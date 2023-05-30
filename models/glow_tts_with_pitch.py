# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.utils.data
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from utils.helpers import (
    log_audio_to_tb,
    plot_alignment_to_numpy,
    plot_spectrogram_to_numpy,
    process_batch,
)
from utils.glow_tts_loss import GlowTTSLoss
from utils.data import load_speaker_emb
from nemo.collections.tts.losses.fastpitchloss import PitchLoss
from nemo.collections.tts.models.base import SpectrogramGenerator

from modules.glow_tts_with_pitch import GlowTTSModule
from modules.glow_tts_modules.glow_tts_submodules import sequence_mask

from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    AcousticEncodedRepresentation,
    LengthsType,
    MelSpectrogramType,
    TokenIndex,
    RegressionValuesType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


@dataclass
class GlowTTSConfig:
    encoder: Dict[Any, Any] = MISSING
    decoder: Dict[Any, Any] = MISSING
    parser: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None
    test_ds: Optional[Dict[Any, Any]] = None


class GlowTTSModel(SpectrogramGenerator):
    """
    GlowTTS model used to generate spectrograms from text
    Consists of a text encoder and an invertible spectrogram decoder
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        # Setup normalizer
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)
        
        # Setup vocabulary (=tokenizer) and input_fft_kwargs (supported only with self.learn_alignment=True)
        input_fft_kwargs = {}

        self.vocab = None
        
        self.ds_class_name = cfg.train_ds.dataset._target_.split(".")[-1]
        
        if self.ds_class_name == "TTSDataset":
            self._setup_tokenizer(cfg)
            assert self.vocab is not None
            input_fft_kwargs["n_embed"] = len(self.vocab.tokens)
            input_fft_kwargs["padding_idx"] = self.vocab.pad

        self._parser = None
        
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(GlowTTSConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.preprocessor = instantiate(self._cfg.preprocessor)
        
        # if self._cfg.parser.get("add_blank"):
        self._cfg.encoder.n_vocab = len(self.vocab.tokens)
        encoder = instantiate(self._cfg.encoder)
        decoder = instantiate(self._cfg.decoder)
        pitch_stats = None
        unvoiced_value = 0.0
        if "pitch_predictor" in self._cfg:
            pitch_predictor = instantiate(self._cfg.pitch_predictor)
            # inject into encoder
            if self._cfg.get("unvoiced_value") is not None:
                unvoiced_value = self._cfg.get("unvoiced_value")
            elif self._cfg.get("use_normalized_pitch"):
                # compute unoviced value
                
                pitch_mean, pitch_std = self._cfg.get("pitch_mean", 0.0), self._cfg.get("pitch_std", 1.0)
                pitch_mean = pitch_mean or 0.0
                pitch_std = pitch_std or 1.0
                unvoiced_value = -pitch_mean/pitch_std
                pitch_stats = {"pitch_mean": pitch_mean, "pitch_std": pitch_std,
                            #    "pitch_fmin": self._cfg.get("pitch_fmin"),
                               }
        else:
            pitch_predictor = None
        self.glow_tts = GlowTTSModule(
            encoder,
            decoder,
            pitch_predictor,
            n_speakers=cfg.n_speakers,
            gin_channels=cfg.gin_channels,
            use_external_speaker_emb=cfg.get("use_external_speaker_emb", False),
            use_stoch_dur_pred=cfg.get("use_stoch_dur_pred", False),
            use_stoch_pitch_pred=cfg.get("use_stoch_pitch_pred", False),
            unvoiced_value = unvoiced_value,
            use_log_pitch = cfg.get("use_log_pitch", False),
            use_normalized_pitch = cfg.get("use_normalized_pitch", False),
            use_frame_emb_for_pitch = cfg.get("use_frame_emb_for_pitch", False),
            pitch_stats=pitch_stats,
        )
        self.loss = GlowTTSLoss()
        if pitch_predictor is not None:
            self.pitch_loss_scale = cfg.get("pitch_loss_scale", 0.1)
            self.pitch_loss_fn = PitchLoss(loss_scale=self.pitch_loss_scale)
            
        else:
            self.pitch_loss_scale = 0.0
            self.pitch_loss_fn = None
        
        
    def parse(self, str_input: str, normalize=True) -> torch.tensor:
        if str_input[-1] not in [".", "!", "?"]:
            str_input = str_input + "."
        
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")

        if normalize and self.text_normalizer_call is not None:
            str_input = self.text_normalizer_call(str_input, **self.text_normalizer_call_kwargs)

        eval_phon_mode = contextlib.nullcontext()
        if hasattr(self.vocab, "set_phone_prob"):
            eval_phon_mode = self.vocab.set_phone_prob(prob=1.0)

        # Disable mixed g2p representation if necessary
        with eval_phon_mode:
            tokens = self.parser(str_input)

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x
    
    @property
    def parser(self):
        if self._parser is not None:
            return self._parser

        ds_class_name = self._cfg.train_ds.dataset._target_.split(".")[-1]

        if ds_class_name == "TTSDataset":
            self._parser = self.vocab.encode
        else:
            raise ValueError(f"Unknown dataset class: {ds_class_name}")

        return self._parser

    @typecheck(
        input_types={
            "x": NeuralType(("B", "T"), TokenIndex()),
            "x_lengths": NeuralType(("B"), LengthsType()),
            "y": NeuralType(("B", "D", "T"), MelSpectrogramType(), optional=True),
            "y_lengths": NeuralType(("B"), LengthsType(), optional=True),
            "gen": NeuralType(optional=True),
            "noise_scale": NeuralType(optional=True),
            "length_scale": NeuralType(optional=True),
            "speaker": NeuralType(("B"), TokenIndex(), optional=True),
            "speaker_embeddings": NeuralType(
                ("B", "D"), AcousticEncodedRepresentation(), optional=True
            ),
            "stoch_dur_noise_scale": NeuralType(optional=True),
            "stoch_pitch_noise_scale": NeuralType(optional=True),
            "pitch_scale": NeuralType(optional=True),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
        }
    )
    
    def forward(
        self,
        *,
        x,
        x_lengths,
        y=None,
        y_lengths=None,
        speaker=None,
        gen=False,
        noise_scale=0.0,
        length_scale=1.0,
        speaker_embeddings=None,
        stoch_dur_noise_scale=1.0,
        stoch_pitch_noise_scale=1.0,
        pitch_scale=0.0,
        pitch=None,
    ):
        if gen:
            return self.glow_tts.generate_spect(
                text=x,
                text_lengths=x_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                speaker=speaker,
                speaker_embeddings=speaker_embeddings,
                stoch_dur_noise_scale=stoch_dur_noise_scale,
                stoch_pitch_noise_scale=stoch_pitch_noise_scale,
                pitch_scale=pitch_scale,
            )
        else:
            return self.glow_tts(
                text=x,
                text_lengths=x_lengths,
                spect=y,
                spect_lengths=y_lengths,
                speaker=speaker,
                speaker_embeddings=speaker_embeddings,
                pitch=pitch,
            )

    def step(
        self,
        y,
        y_lengths,
        x,
        x_lengths,
        speaker,
        speaker_embeddings,
        pitch,
    ):
        
        z, y_m, y_logs, logdet, logw, logw_, y_lengths, attn, stoch_dur_loss, pitch, pitch_pred, pitch_loss = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            speaker=speaker,
            speaker_embeddings=speaker_embeddings,
            pitch=pitch,
        )

        l_mle, l_length, logdet = self.loss(
            z=z,
            y_m=y_m,
            y_logs=y_logs,
            logdet=logdet,
            logw=logw,
            logw_=logw_,
            x_lengths=x_lengths,
            y_lengths=y_lengths,
            stoch_dur_loss=stoch_dur_loss,
        )

        if self.pitch_loss_fn is not None:
            if pitch_loss is None:
                pitch_loss = self.pitch_loss_fn(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=x_lengths)
            else:
                pitch_loss = self.pitch_loss_scale*pitch_loss
        
        if pitch_loss is None:
            loss = sum([l_mle, l_length])
            pitch_loss = torch.tensor([0.0]).to(device=l_mle.device)
        else:
            loss = sum([l_mle, l_length, pitch_loss])
            
        
        return l_mle, l_length, logdet, loss, attn, pitch_loss

    def training_step(self, batch, batch_idx):

        batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
        y = batch_dict.get("audio")
        y_lengths = batch_dict.get("audio_lens")
        x = batch_dict.get("text")
        x_lengths = batch_dict.get("text_lens")
        attn_prior = batch_dict.get("align_prior_matrix", None)
        pitch = batch_dict.get("pitch", None)
        energy = batch_dict.get("energy", None)
        speaker = batch_dict.get("speaker_id", None)
        speaker_embeddings = batch_dict.get("speaker_emb", None)

        y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

        l_mle, l_length, logdet, loss, _, pitch_loss = self.step(
            y, y_lengths, x, x_lengths, speaker, speaker_embeddings, pitch
        )

        output = {
            "loss": loss,  # required
            "progress_bar": {"l_mle": l_mle, "l_length": l_length, "logdet": logdet},
            "log": {"loss": loss, "l_mle": l_mle, "l_length": l_length, "logdet": logdet, "pitch_loss": pitch_loss},
        }

        return output

    @torch.no_grad()
    def compute_likelihood(self, batch_dict):
        y = batch_dict.get("audio")
        y_lengths = batch_dict.get("audio_lens")
        x = batch_dict.get("text")
        x_lengths = batch_dict.get("text_lens")
        attn_prior = batch_dict.get("align_prior_matrix", None)
        pitch = batch_dict.get("pitch", None)
        energy = batch_dict.get("energy", None)
        speaker = batch_dict.get("speaker_id", None)
        speaker_embeddings = batch_dict.get("speaker_emb", None)

        y, y_lengths, x, x_lengths, speaker_embeddings = (
            y.to(self.device),
            y_lengths.to(self.device),
            x.to(self.device),
            x_lengths.to(self.device),
            speaker_embeddings.to(self.device),
        )
        y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)
        
        z, y_m, y_logs, logdet, logw, logw_, y_lengths, attn, stoch_dur_loss, pitch, pitch_pred, pitch_loss = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            speaker=speaker,
            speaker_embeddings=speaker_embeddings,
            pitch=pitch,
        )
        
        l_mle_normal = 0.5 * math.log(2 * math.pi) + (
            torch.sum(y_logs) + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2)
        ) / (torch.sum(y_lengths) * z.shape[1])
        
        return l_mle_normal

    def validation_step(self, batch, batch_idx):
        
        batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
        y = batch_dict.get("audio")
        y_lengths = batch_dict.get("audio_lens")
        x = batch_dict.get("text")
        x_lengths = batch_dict.get("text_lens")
        attn_prior = batch_dict.get("align_prior_matrix", None)
        pitch = batch_dict.get("pitch", None)
        energy = batch_dict.get("energy", None)
        speaker = batch_dict.get("speaker_id", None)
        speaker_embeddings = batch_dict.get("speaker_emb", None)

        y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

        l_mle, l_length, logdet, loss, attn, pitch_loss = self.step(
            y,
            y_lengths,
            x,
            x_lengths,
            speaker,
            speaker_embeddings,
            pitch,
        )

        y_gen, attn_gen = self(
            x=x,
            x_lengths=x_lengths,
            gen=True,
            speaker=speaker,
            speaker_embeddings=speaker_embeddings,
            pitch=None, # use predicted pitch
            noise_scale=0.667,
        )

        return {
            "loss": loss,
            "l_mle": l_mle,
            "l_length": l_length,
            "logdet": logdet,
            "y": y,
            "y_gen": y_gen,
            "x": x,
            "attn": attn,
            "attn_gen": attn_gen,
            "progress_bar": {"l_mle": l_mle, "l_length": l_length, "logdet": logdet},
            "pitch_loss": pitch_loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_mle = torch.stack([x["l_mle"] for x in outputs]).mean()
        avg_length_loss = torch.stack([x["l_length"] for x in outputs]).mean()
        avg_logdet = torch.stack([x["logdet"] for x in outputs]).mean()
        avg_pitch_loss = torch.stack([x["pitch_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_mle": avg_mle,
            "val_length_loss": avg_length_loss,
            "val_logdet": avg_logdet,
            "val_pitch_loss": avg_pitch_loss,
        }
        if self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break

            separated_tokens = self.vocab.decode(outputs[0]["x"][0])

            tb_logger.add_text("separated tokens", separated_tokens, self.global_step)
            tb_logger.add_image(
                "real_spectrogram",
                plot_spectrogram_to_numpy(outputs[0]["y"][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            tb_logger.add_image(
                "generated_spectrogram",
                plot_spectrogram_to_numpy(outputs[0]["y_gen"][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            tb_logger.add_image(
                "alignment_for_real_sp",
                plot_alignment_to_numpy(outputs[0]["attn"][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            tb_logger.add_image(
                "alignment_for_generated_sp",
                plot_alignment_to_numpy(outputs[0]["attn_gen"][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            log_audio_to_tb(tb_logger, outputs[0]["y"][0], "true_audio_gf", self.global_step)
            log_audio_to_tb(
                tb_logger, outputs[0]["y_gen"][0], "generated_audio_gf", self.global_step
            )
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss, "log": tensorboard_logs}
    
    def _setup_normalizer(self, cfg):
        if "text_normalizer" in cfg:
            normalizer_kwargs = {}

            if "whitelist" in cfg.text_normalizer:
                normalizer_kwargs["whitelist"] = self.register_artifact(
                    'text_normalizer.whitelist', cfg.text_normalizer.whitelist
                )

            try:
                self.normalizer = instantiate(cfg.text_normalizer, **normalizer_kwargs)
            except Exception as e:
                logging.error(e)
                raise ImportError(
                    "`pynini` not installed, please install via NeMo/nemo_text_processing/pynini_install.sh"
                )

            self.text_normalizer_call = self.normalizer.normalize
            if "text_normalizer_call_kwargs" in cfg:
                self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs
                
    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}

        if "phoneme_dict" in cfg.text_tokenizer:
            text_tokenizer_kwargs["phoneme_dict"] = self.register_artifact(
                "text_tokenizer.phoneme_dict", cfg.text_tokenizer.phoneme_dict,
            )
        if "heteronyms" in cfg.text_tokenizer:
            text_tokenizer_kwargs["heteronyms"] = self.register_artifact(
                "text_tokenizer.heteronyms", cfg.text_tokenizer.heteronyms,
            )

        if "g2p" in cfg.text_tokenizer:
            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.vocab = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)
        
    def _setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")
        
        if cfg.dataset._target_ == "utils.data.TTSDataset":
            phon_mode = contextlib.nullcontext()
            if hasattr(self.vocab, "set_phone_prob"):
                phon_mode = self.vocab.set_phone_prob(prob=None if name == "val" else self.vocab.phoneme_probability)

            with phon_mode:
                print("I got here!!!!!!!", self.vocab)
                dataset = instantiate(
                    cfg.dataset,
                    text_normalizer=self.normalizer,
                    text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
                    text_tokenizer=self.vocab,
                )
        else:
            dataset = instantiate(cfg.dataset)

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, name="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, shuffle_should_be=False, name="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, shuffle_should_be=False, name="test")

    
    def generate_spectrogram(
        self,
        tokens: "torch.tensor",
        noise_scale: float = 0.0,
        length_scale: float = 1.0,
        speaker: int = None,
        speaker_embeddings: "torch.tensor" = None,
        stoch_dur_noise_scale: float = 1.0,
        stoch_pitch_noise_scale: float = 1.0,
        pitch_scale: float = 0.0,
    ) -> torch.tensor:

        self.eval()

        token_len = torch.tensor([tokens.shape[1]]).to(self.device)

        if isinstance(speaker, int):
            speaker = torch.tensor([speaker]).to(self.device)
        else:
            speaker = None
        
        if speaker_embeddings is not None:
            speaker_embeddings = speaker_embeddings.to(self.device)

        spect, _ = self(
            x=tokens,
            x_lengths=token_len,
            speaker=speaker,
            gen=True,
            noise_scale=noise_scale,
            length_scale=length_scale,
            speaker_embeddings=speaker_embeddings,
            stoch_dur_noise_scale=stoch_dur_noise_scale,
            stoch_pitch_noise_scale=stoch_pitch_noise_scale,
            pitch_scale=pitch_scale,
        )

        return spect

    @torch.no_grad()
    def generate_spectrogram_with_mas(
        self,
        batch,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        external_speaker: int = None,
        external_speaker_embeddings: "torch.tensor" = None,
        gen: bool = False,
        randomize_speaker=False,
    ) -> torch.tensor:
        """Forced aligned generation of synthetic mel-spectrogram of mel-spectrograms."""
        # y is audio or melspec, x is text token ids
        self.eval()
        
        batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
        y = batch_dict.get("audio")
        y_lengths = batch_dict.get("audio_lens")
        x = batch_dict.get("text")
        x_lengths = batch_dict.get("text_lens")
        attn_prior = batch_dict.get("align_prior_matrix", None)
        pitch = batch_dict.get("pitch", None)
        energy = batch_dict.get("energy", None)
        speaker = batch_dict.get("speaker_id", None)
        speaker_embeddings = batch_dict.get("speaker_emb", None)


        y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

        if len(speaker_embeddings.shape) == 1:
            speaker_embeddings = speaker_embeddings.unsqueeze(0)
        speaker = None
        if speaker_embeddings is not None:
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings).unsqueeze(-1).to(self.device)

        y, y_lengths, x, x_lengths, speaker_embeddings = (
            y.to(self.device),
            y_lengths.to(self.device),
            x.to(self.device),
            x_lengths.to(self.device),
            speaker_embeddings,
        )
        (z, y_m, y_logs, logdet, log_durs_predicted, log_durs_extracted, \
            spect_lengths, attn, stoch_dur_loss, pitch, pitch_pred, pitch_pred_loss
        ) = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            speaker=speaker,
            gen=gen,
            speaker_embeddings=speaker_embeddings.squeeze(-1),
            pitch=pitch,
        )
        
        y_max_length = z.size(2)

        y_mask = torch.unsqueeze(sequence_mask(spect_lengths, y_max_length), 1)

        # predicted aligned feature
        z = (y_m + torch.exp(y_logs) * torch.randn_like(y_m) * noise_scale) * y_mask

        if external_speaker_embeddings is not None:
            external_speaker_embeddings = torch.nn.functional.normalize(
                external_speaker_embeddings
            ).unsqueeze(-1)

        if randomize_speaker:
            # use a random speaker embedding to decode utterance
            idx = torch.randperm(speaker_embeddings.shape[0])
            speaker_embeddings = speaker_embeddings[idx].view(speaker_embeddings.size())
            if len(speaker_embeddings.shape) == 2:
                speaker_embeddings = speaker_embeddings.unsqueeze(-1).to(self.device)

        # invert with same or different speaker through the decoder
        y_pred, _ = self.glow_tts.decoder(
            spect=z.to(self.device),
            spect_mask=y_mask.to(self.device),
            speaker_embeddings=external_speaker_embeddings or speaker_embeddings,
            reverse=True, pitch=pitch,
        )
        return y_pred, spect_lengths
    
    
    @classmethod
    def list_available_models(cls) -> "List[PretrainedModelInfo]":
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_glowtts",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_glowtts/versions/1.0.0rc1/files/tts_en_glowtts.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate female English voices with an American accent.",
            class_=cls,
            aliases=["GlowTTS-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models