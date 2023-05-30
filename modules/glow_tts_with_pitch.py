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

# MIT License
#
# Copyright (c) 2020 Jaehyeon Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math

import torch
import torch.nn.functional as F
from torch import nn

from modules.glow_tts_modules import glow_tts_submodules_with_pitch as glow_tts_submodules

from nemo.collections.tts.helpers.helpers import regulate_len
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    AcousticEncodedRepresentation,
    IntType,
    LengthsType,
    LogDeterminantType,
    MaskType,
    MelSpectrogramType,
    NormalDistributionLogVarianceType,
    NormalDistributionMeanType,
    NormalDistributionSamplesType,
    SequenceToSequenceAlignmentType,
    TokenIndex,
    TokenLogDurationType,
    EncodedRepresentation,
    TokenDurationType,
    RegressionValuesType,
)
from nemo.core.neural_types.neural_type import NeuralType
from torch.cuda.amp import autocast

class TextEncoder(NeuralModule):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        filter_channels_dp: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        window_size: int,
        mean_only: bool = False,
        prenet: bool = False,
        gin_channels: int = 0,
        use_stoch_dur_pred = False,
        n_flows=4, # no. of flows of stochastic predictor
    ):
        """
        GlowTTS text encoder. Takes in the input text tokens and produces prior distribution statistics for the latent
        representation corresponding to each token, as well as the log durations (the Duration Predictor is also part of
         this module).
        Architecture is similar to Transformer TTS with slight modifications.
        Args:
            n_vocab (int): Number of tokens in the vocabulary
            out_channels (int): Latent representation channels
            hidden_channels (int): Number of channels in the intermediate representations
            filter_channels (int): Number of channels for the representations in the feed-forward layer
            filter_channels_dp (int): Number of channels for the representations in the duration predictor
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            kernel_size (int): Kernels size for the feed-forward layer
            p_dropout (float): Dropout probability
            mean_only (bool): Return zeros for logs if true
            prenet (bool): Use an additional network before the transformer modules
            gin_channels (int): Number of channels in speaker embeddings
        """
        super().__init__()

        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.prenet = prenet
        self.mean_only = mean_only

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if prenet:
            self.pre = glow_tts_submodules.ConvReluNorm(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.1,
            )

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                glow_tts_submodules.AttentionBlock(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    window_size=window_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(glow_tts_submodules.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                glow_tts_submodules.FeedForwardNetwork(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(glow_tts_submodules.LayerNorm(hidden_channels))

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
            
        self.use_stoch_dur_pred = use_stoch_dur_pred
        if use_stoch_dur_pred:
            self.proj_w = glow_tts_submodules.StochasticDurationPredictor(
                hidden_channels, filter_channels_dp, kernel_size, p_dropout, n_flows, gin_channels,
            )
        else:
            self.proj_w = glow_tts_submodules.DurationPredictor(
                hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout,
            )
        self.unvoiced_value = None

    @property
    def input_types(self):
        return {
            "text": NeuralType(("B", "T"), TokenIndex()),
            "text_lengths": NeuralType(("B",), LengthsType()),
            "speaker_embeddings": NeuralType(
                ("B", "D", "T"), AcousticEncodedRepresentation(), optional=True
            ),
        }

    @property
    def output_types(self):
        return {
            "x_m": NeuralType(("B", "D", "T"), NormalDistributionMeanType()),
            # "x_logs": NeuralType(("B", "D", "T"), NormalDistributionLogVarianceType()),
            # "logw": NeuralType(("B", "T"), TokenLogDurationType()),
            "x_mask": NeuralType(("B", "D", "T"), MaskType()),
        }

    @typecheck()
    def forward(self, *, text, text_lengths, speaker_embeddings,):

        x = self.emb(text) * math.sqrt(self.hidden_channels)  # [b, t, h]

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(glow_tts_submodules.sequence_mask(text_lengths, x.size(2)), 1).to(
            x.dtype
        )

        if self.prenet:
            x = self.pre(x, x_mask)

        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        
        return x, x_mask


class FlowSpecDecoder(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_blocks: int,
        n_layers: int,
        p_dropout: float = 0.0,
        n_split: int = 4,
        n_sqz: int = 2,
        sigmoid_scale: bool = False,
        gin_channels: int = 0,
    ):
        """
        Flow-based invertible decoder for GlowTTS. Converts spectrograms to latent representations and back.
        Args:
            in_channels (int): Number of channels in the input spectrogram
            hidden_channels (int): Number of channels in the intermediate representations
            kernel_size (int): Kernel size in the coupling blocks
            dilation_rate (int): Dilation rate in the WaveNet-like blocks
            n_blocks (int): Number of flow blocks
            n_layers (int): Number of layers within each coupling block
            p_dropout (float): Dropout probability
            n_split (int): Group size for the invertible convolution
            n_sqz (int): The rate by which the spectrograms are squeezed before applying the flows
            sigmoid_scale (bool): Apply sigmoid to logs in the coupling blocks
        """
        super().__init__()

        self.n_sqz = n_sqz

        self.flows = nn.ModuleList()
        for _ in range(n_blocks):
            self.flows.append(glow_tts_submodules.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(
                glow_tts_submodules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split)
            )
            self.flows.append(
                glow_tts_submodules.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                    gin_channels=gin_channels,
                    n_sqz=n_sqz,
                )
            )

    @property
    def input_types(self):
        return {
            "spect": NeuralType(("B", "D", "T"), optional=True ), #MelSpectrogramType()),
            "spect_mask": NeuralType(("B", "D", "T"), MaskType()),
            "speaker_embeddings": NeuralType(
                ("B", "D", "T"), AcousticEncodedRepresentation(), optional=True
            ),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "reverse": NeuralType(elements_type=IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "z": NeuralType(("B", "D", "T"), NormalDistributionSamplesType()),
            "logdet_tot": NeuralType(("B",), LogDeterminantType()),
        }

    @typecheck()
    def forward(self, *, spect, spect_mask, speaker_embeddings=None, pitch=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        x = spect
        x_mask = spect_mask

        if self.n_sqz > 1:
            x, x_mask = self.squeeze(x, x_mask, self.n_sqz)

        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=speaker_embeddings, 
                              pitch=pitch, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=speaker_embeddings, 
                              pitch=pitch, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = self.unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def squeeze(self, x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        t = (t // n_sqz) * n_sqz
        x = x[:, :, :t]
        x_sqz = x.view(b, c, t // n_sqz, n_sqz)
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

        if x_mask is not None:
            x_mask = x_mask[:, :, n_sqz - 1 :: n_sqz]
        else:
            x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
        return x_sqz * x_mask, x_mask

    def unsqueeze(self, x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
        x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
        else:
            x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
        return x_unsqz * x_mask, x_mask

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()


class GlowTTSModule(NeuralModule):
    def __init__(
        self,
        encoder_module: NeuralModule,
        decoder_module: NeuralModule,
        pitch_predictor = None,
        n_speakers: int = 1,
        gin_channels: int = 0,
        use_external_speaker_emb: bool = False,
        use_stoch_dur_pred=False,
        use_stoch_pitch_pred=False,
        unvoiced_value = 0.0,
        use_log_pitch = False,
        use_normalized_pitch = False,
        use_frame_emb_for_pitch = False,
        pitch_stats = None,
    ):
        """
        Main GlowTTS module. Contains the encoder and decoder.
        Args:
            encoder_module (NeuralModule): Text encoder for predicting latent distribution statistics
            decoder_module (NeuralModule): Invertible spectrogram decoder
            n_speakers (int): Number of speakers
            gin_channels (int): Channels in speaker embeddings
        """
        super().__init__()

        self.encoder = encoder_module
        self.decoder = decoder_module
        self.pitch_predictor = pitch_predictor
        self.n_speakers = n_speakers

        if n_speakers > 1 and use_external_speaker_emb is False:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            
        self.use_stoch_dur_pred = use_stoch_dur_pred
        self.use_stoch_pitch_pred = use_stoch_pitch_pred
        self.unvoiced_value = unvoiced_value
        self.use_log_pitch = use_log_pitch
        self.use_normalized_pitch = use_normalized_pitch
        self.use_frame_emb_for_pitch = use_frame_emb_for_pitch
        if self.use_normalized_pitch:
            assert pitch_stats is not None
            self.pitch_mean = pitch_stats.get("pitch_mean")
            self.pitch_std = pitch_stats.get("pitch_std")
            # self.unvoiced_value = pitch_stats.get("pitch_fmin")
    @property
    def input_types(self):
        return {
            "text": NeuralType(("B", "T"), TokenIndex()),
            "text_lengths": NeuralType(("B"), LengthsType()),
            "spect": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "spect_lengths": NeuralType(("B"), LengthsType()),
            "speaker": NeuralType(("B"), IntType(), optional=True),
            "speaker_embeddings": NeuralType(
                ("B", "D"), AcousticEncodedRepresentation(), optional=True
            ),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "z": NeuralType(("B", "D", "T"), NormalDistributionSamplesType()),
            "y_m": NeuralType(("B", "D", "T"), NormalDistributionMeanType()),
            "y_logs": NeuralType(("B", "D", "T"), NormalDistributionLogVarianceType()),
            "logdet": NeuralType(("B"), LogDeterminantType()),
            "log_durs_predicted": NeuralType(("B", "T"), TokenLogDurationType()),
            "log_durs_extracted": NeuralType(("B", "T"), TokenLogDurationType()),
            "spect_lengths": NeuralType(("B"), LengthsType()),
            "attn": NeuralType(("B", "T", "T"), SequenceToSequenceAlignmentType()),
            "stoch_dur_loss": NeuralType(optional=True),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "pitch_pred": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "pitch_pred_loss": NeuralType(optional=True),
            
        }

    @typecheck()
    def forward(
        self,
        *,
        text,
        text_lengths,
        spect,
        spect_lengths,
        speaker=None,
        speaker_embeddings=None,
        pitch=None,
    ):

        if speaker_embeddings is not None:
            speaker_embeddings = F.normalize(speaker_embeddings).unsqueeze(-1)
        elif speaker is not None and self.n_speakers > 1:
            speaker_embeddings = F.normalize(self.emb_g(speaker)).unsqueeze(-1)  # [b, h]

        x, x_mask = self.encoder(
            text=text, text_lengths=text_lengths, speaker_embeddings=speaker_embeddings,
        )
        x_m = self.encoder.proj_m(x) * x_mask
        if not self.encoder.mean_only:
            x_logs = self.encoder.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        y_max_length = spect.size(2)
        y_max_length = (y_max_length // self.decoder.n_sqz) * self.decoder.n_sqz
        spect = spect[:, :, :y_max_length]

        spect_lengths = (spect_lengths // self.decoder.n_sqz) * self.decoder.n_sqz

        y_mask = torch.unsqueeze(
            glow_tts_submodules.sequence_mask(spect_lengths, y_max_length), 1
        ).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        
        pitch_norm1 = None
        if pitch is not None:
            # ensure length of pitch is same as spectrogram
            pitch = pitch[:, :y_max_length]

            if self.use_log_pitch:
                pitch_mask = (pitch == 0.0)
                pitch_norm1 = torch.log(torch.clamp(pitch, min=torch.finfo(pitch.dtype).tiny))
                pitch_norm1[pitch_mask] = 0.0
            elif self.use_normalized_pitch:
                pitch_mask = (pitch == 0.0)
                pitch_norm1 = (pitch - self.pitch_mean)
                pitch_norm1[pitch_mask] = 0.0
                pitch_norm1 = pitch_norm1/self.pitch_std
                
            else:
                raise Exception("One normalization method has to be set between log or norm")
            
        z, logdet = self.decoder(
            spect=spect, spect_mask=y_mask, speaker_embeddings=speaker_embeddings, 
            pitch=pitch_norm1, reverse=False,
        )

        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]
                              ).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(
                x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2)
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul(
                (x_m * x_s_sq_r).transpose(1, 2), z
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

            attn = (
                glow_tts_submodules.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
            ).squeeze(1)

        y_m = torch.matmul(x_m, attn)
        y_logs = torch.matmul(x_logs, attn)

        durs_extracted = torch.sum(attn, -1)
        log_durs_extracted = torch.log(1e-8 + durs_extracted) * x_mask.squeeze()

        if self.use_stoch_dur_pred:
            w = attn.sum(2).unsqueeze(1)
            stoch_dur_loss, log_durs_predicted = self.encoder.proj_w(spect=x, x_mask=x_mask, 
                                                       w=w, g=speaker_embeddings,)
        else:
            if speaker_embeddings is not None:
                g_exp = speaker_embeddings.expand(-1, -1, x.size(-1))
                x_dp = torch.cat([torch.detach(x), g_exp], 1)
            else:
                x_dp = torch.detach(x)
            log_durs_predicted = self.encoder.proj_w(spect=x_dp, mask=x_mask,)
            stoch_dur_loss = None
        
        # Predict pitch for each token
        # Pitch during training is per spectrogram frame, but during inference, it should be per character
        pitch_pred = None
        pitch_pred_loss = None
        pitch_norm = None
        if pitch is not None and self.pitch_predictor is not None:
            # expand embeddings if required
            if self.use_frame_emb_for_pitch:
                x_pitch = torch.matmul(x, attn)
                if self.use_log_pitch:
                    # normalize pitch
                    pitch_mask2 = (pitch == 0.0)
                    pitch_norm = torch.log(torch.clamp(pitch, min=torch.finfo(pitch.dtype).tiny))
                    pitch_norm[pitch_mask2] = 0.0
                elif self.use_normalized_pitch:
                    pitch_mask2 = (pitch == 0.0)
                    pitch_norm = (pitch - self.pitch_mean)
                    pitch_norm[pitch_mask2] = 0.0
                    pitch_norm = pitch_norm / self.pitch_std
                    pitch_norm = pitch_norm*y_mask.squeeze()
                pitch_norm = pitch_norm.squeeze(1)
                
                if self.use_stoch_pitch_pred:
                    pitch_pred_loss, pitch_pred = self.pitch_predictor(x_pitch, y_mask, w=pitch_norm.unsqueeze(1), g=speaker_embeddings,)
                    if pitch_pred:
                        pitch_pred = pitch_pred.squeeze(1)
                else: # either normal prediction or no pitch pred
                    if hasattr(self.pitch_predictor, "gin_channels"):
                        pitch_pred = self.pitch_predictor(x_pitch, y_mask, speaker_embeddings)
                    else:
                        if speaker_embeddings is not None:
                            speaker_cond = speaker_embeddings.expand(-1, -1, x_pitch.size(-1))
                        x_p = torch.cat([torch.detach(x_pitch), speaker_cond], 1)
                        pitch_pred = self.pitch_predictor(x_p, y_mask)
                    pitch_pred_loss = None
                
            else:
                x_pitch = x.detach()
                
                # # concat speaker embedding if required
                # if not self.use_stoch_pitch_pred and speaker_embeddings is not None:
                #     x_p = torch.cat([torch.detach(x_pitch), g_exp], 1)
                # else:
                #     x_p = torch.detach(x_pitch)
            
                # average pitch if predicting on phonemes
                if pitch.shape[-1] != x_pitch.shape[-1]:                  
                    durs_extracted = durs_extracted * x_mask.squeeze()
                    
                    if self.use_normalized_pitch:
                        # avg feature func only works when unvoiced regions are set to zero
                        # pitch_norm = pitch.detach().clone()
                        # pitch_norm[pitch_norm == self.unvoiced_value] = 0.0
                        
                        pitch_norm = average_features(pitch.unsqueeze(1), durs_extracted)
                        # pitch_norm[pitch_norm == 0.0] = self.unvoiced_value
                        pitch_norm = pitch_norm*x_mask.squeeze()
                        
                    else: # log prediction does not need it as it was already set to zero
                        # now we want to average the original pitch values
                        pitch_norm = average_features(pitch.unsqueeze(1), durs_extracted)
                else:
                    pitch_norm = pitch
                
                if self.use_log_pitch:
                    # normalize pitch
                    pitch_mask2 = (pitch_norm == 0.0)
                    pitch_norm = torch.log(torch.clamp(pitch_norm, min=torch.finfo(pitch_norm.dtype).tiny))
                    pitch_norm[pitch_mask2] = 0.0
                elif self.use_normalized_pitch:
                    pitch_mask2 = (pitch_norm == 0.0)
                    pitch_norm = (pitch_norm - self.pitch_mean)
                    pitch_norm[pitch_mask2] = 0.0
                    pitch_norm = pitch_norm/self.pitch_std
                    # pitch_norm[pitch_mask2] = self.unvoiced_value
                    pitch_norm = pitch_norm*x_mask
                    # pitch_log2 = pitch_log2*x_mask
                pitch_norm = pitch_norm.squeeze(1)
            
                if self.use_stoch_pitch_pred:
                    pitch_pred_loss, pitch_pred = self.pitch_predictor(x_pitch, x_mask, w=pitch_norm.unsqueeze(1), g=speaker_embeddings,)
                    if pitch_pred:
                        pitch_pred = pitch_pred.squeeze(1)
                else: # either normal prediction or no pitch pred
                    if hasattr(self.pitch_predictor, "gin_channels"):
                        pitch_pred = self.pitch_predictor(x_pitch, x_mask, speaker_embeddings)
                    else:
                        x_p = torch.cat([torch.detach(x_pitch), g_exp], 1)
                        pitch_pred = self.pitch_predictor(x_p, x_mask)
                    pitch_pred_loss = None

        return z, y_m, y_logs, logdet, log_durs_predicted, log_durs_extracted, \
            spect_lengths, attn, stoch_dur_loss, pitch_norm, pitch_pred, pitch_pred_loss

    @typecheck(
        input_types={
            "text": NeuralType(("B", "T"), TokenIndex()),
            "text_lengths": NeuralType(("B",), LengthsType()),
            "noise_scale": NeuralType(optional=True),
            "length_scale": NeuralType(optional=True),
            "speaker": NeuralType(("B"), IntType(), optional=True),
            "speaker_embeddings": NeuralType(
                ("B", "D"), AcousticEncodedRepresentation(), optional=True
            ),
            "stoch_dur_noise_scale": NeuralType(optional=True),
            "stoch_pitch_noise_scale": NeuralType(optional=True),
            "pitch_scale": NeuralType(optional=True),
        },
        output_types={
            "y": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "attn": NeuralType(("B", "T", "T"), SequenceToSequenceAlignmentType()),
        },
    )
    
    def generate_spect(
        self,
        *,
        text,
        text_lengths,
        noise_scale=0.667,
        length_scale=1.0,
        speaker=None,
        speaker_embeddings=None,
        stoch_dur_noise_scale=1.0,
        stoch_pitch_noise_scale=1.0,
        pitch_scale=0.0,
    ):

        if speaker_embeddings is not None:
            # using external speaker embedding
            speaker_embeddings = F.normalize(speaker_embeddings).unsqueeze(-1)
        elif speaker is not None and self.n_speakers > 1:
            # using global style tokens
            speaker_embeddings = F.normalize(self.emb_g(speaker)).unsqueeze(-1)  # [b, h]

        x, x_mask = self.encoder(
            text=text, text_lengths=text_lengths, speaker_embeddings=speaker_embeddings,
        )
        
        x_m = self.encoder.proj_m(x) * x_mask
        if not self.encoder.mean_only:
            x_logs = self.encoder.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        # # deal with duration here too
        # g_exp = speaker_embeddings.expand(-1, -1, x.size(-1)) if speaker_embeddings is not None else None
        # if not self.use_stoch_dur_pred and g_exp is not None:
        #     #TODO: check the correct implementation for speakers concat or add?
        #     x_dp = torch.cat([torch.detach(x), g_exp], 1)
        # else:
        #     x_dp = torch.detach(x)

        if self.use_stoch_dur_pred:
            _, log_durs_predicted = self.encoder.proj_w(spect=torch.detach(x), x_mask=x_mask, 
                                        w=None, g=speaker_embeddings, noise_scale=stoch_dur_noise_scale,
                                        reverse=True,)
            log_durs_predicted = torch.squeeze(log_durs_predicted, 1)

        else:
            if speaker_embeddings is not None:
                g_exp = speaker_embeddings.expand(-1, -1, x.size(-1))
                x_dp = torch.cat([torch.detach(x), g_exp], 1)
            else:
                x_dp = torch.detach(x)
            log_durs_predicted = self.encoder.proj_w(spect=x_dp, mask=x_mask)
            
        w = torch.exp(log_durs_predicted) * x_mask.squeeze() * length_scale
        
        w_ceil = torch.ceil(w)
        # reduce the silence at the beginning and end of utterances
        # w_ceil[:, 0] = torch.clamp(w_ceil[:, 0], min=0, max=5)
        # w_ceil[:, -1] = torch.clamp(w_ceil[:, -1], min=0, max=5)
        
        w_ceil[:, 0] = 10 #16ms silence for evaluation
        w_ceil[:, -1] = 10 #16ms silence for evaluation
        
        spect_lengths = torch.clamp_min(torch.sum(w_ceil, [1]), 1).long()
        y_max_length = None

        spect_lengths = (spect_lengths // self.decoder.n_sqz) * self.decoder.n_sqz

        y_mask = torch.unsqueeze(
            glow_tts_submodules.sequence_mask(spect_lengths, y_max_length), 1
        ).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = glow_tts_submodules.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))

        y_m = torch.matmul(x_m, attn)
        y_logs = torch.matmul(x_logs, attn)

        z = (y_m + torch.exp(y_logs) * torch.randn_like(y_m) * noise_scale) * y_mask
        
        # if pitch is None:
        # Predict pitch if pitch is not given e.g. during validation
        # if speaker_embeddings is not None:
        #     # speaker_cond = speaker_embeddings.expand(-1, -1, x.size(-1))
        #     x_p = torch.cat([torch.detach(x), g_exp], 1)
        # else:
        #     x_p = torch.detach(x)
        
        if self.pitch_predictor is not None:
            # expand embeddings if required
            if self.use_frame_emb_for_pitch:
                x_pitch = torch.matmul(x, attn)

                if self.use_stoch_pitch_pred:
                    _, pitch = self.pitch_predictor(x_pitch, y_mask, w=None, g=speaker_embeddings, noise_scale=stoch_pitch_noise_scale,
                                                reverse=True,)
                else:
                    if hasattr(self.pitch_predictor, "gin_channels"):
                        pitch = self.pitch_predictor(x_pitch, y_mask, speaker_embeddings)
                    else:
                        if speaker_embeddings is not None:
                            speaker_cond = speaker_embeddings.expand(-1, -1, x_pitch.size(-1))
                        x_p = torch.cat([torch.detach(x_pitch), speaker_cond], 1)
                        pitch = self.pitch_predictor(x_p, y_mask)
                
            else:
                if self.use_stoch_pitch_pred:
                    _, pitch = self.pitch_predictor(x, x_mask, w=None, g=speaker_embeddings, noise_scale=stoch_pitch_noise_scale,
                                                reverse=True,)

                else:
                    if hasattr(self.pitch_predictor, "gin_channels"):
                        pitch = self.pitch_predictor(x, x_mask, speaker_embeddings)
                    else:
                        if speaker_embeddings is not None:
                            # speaker_cond = speaker_embeddings.expand(-1, -1, x.size(-1))
                            x_p = torch.cat([torch.detach(x), g_exp], 1)
                        pitch = self.pitch_predictor(x_p, x_mask)
            
            pitch = pitch.squeeze(1)
            pitch = torch.clamp_min(pitch, 0)

            if pitch.shape[-1] != z.shape[-1]:
                # need to expand predicted pitch to match no of tokens
                durs_predicted = torch.sum(attn, -1) * x_mask.squeeze()
                pitch, _ = regulate_len(durs_predicted, pitch.unsqueeze(-1))
                pitch = pitch.squeeze(-1)
            
            pitch = pitch+pitch_scale
            # pitch[pitch_mask] = 0.0
            pitch = pitch.squeeze(1)
        else:
            pitch = None

        y, _ = self.decoder(
            spect=z, spect_mask=y_mask, speaker_embeddings=speaker_embeddings, 
            reverse=True, pitch=pitch,
        )

        return y, attn


class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = torch.nn.functional.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.dropout(out)
    

class DurationPredictor2(NeuralModule):
    """Predicts a single float per each temporal location. Borrowed from fastpitch"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2, gin_channels=0):
        super(DurationPredictor2, self).__init__()
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, input_size, 1)
        self.layers = torch.nn.Sequential(
            *[
                ConvReLUNorm(
                    input_size if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout
                )
                for i in range(n_layers)
            ]
        )
        self.fc = torch.nn.Linear(filter_size, 1, bias=True)

    @property
    def input_types(self):
        return {
            "enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "enc_mask": NeuralType(('B', 'T', 1), TokenDurationType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T'), EncodedRepresentation()),
        }

    def forward(self, enc, enc_mask, g=None):
        enc = torch.detach(enc)
        if g is not None:
            g = torch.detach(g)
            enc = enc + self.cond(g)
        out = enc * enc_mask
        # out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.layers(out).transpose(1, 2)
        out = self.fc(out) 
        out = out * enc_mask.transpose(1, 2)
        return out.squeeze(-1)


class PitchPredictor(NeuralModule):
    """Predicts a single float per each temporal location. Borrowed from fastpitch"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2, gin_channels=0):
        super(PitchPredictor, self).__init__()
        self.gin_channels = gin_channels
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, input_size, 1)
        self.layers = torch.nn.Sequential(
            *[
                ConvReLUNorm(
                    input_size if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout
                )
                for i in range(n_layers)
            ]
        )
        self.fc = torch.nn.Linear(filter_size, 1, bias=True)

    @property
    def input_types(self):
        return {
            "enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "enc_mask": NeuralType(('B', 'T', 1), TokenDurationType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T'), EncodedRepresentation()),
        }

    def forward(self, enc, enc_mask, g=None):
        enc = torch.detach(enc)
        if g is not None:
            g = torch.detach(g)
            enc = enc + self.cond(g)
        out = enc * enc_mask
        # out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.layers(out).transpose(1, 2)
        out = self.fc(out) 
        out = out * enc_mask.transpose(1, 2)
        return out.squeeze(-1)


class TemporalPredictor(NeuralModule):
    """Predicts a single float per each temporal location. Borrowed from fastpitch"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2):
        super(TemporalPredictor, self).__init__()

        self.layers = torch.nn.Sequential(
            *[
                ConvReLUNorm(
                    input_size if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout
                )
                for i in range(n_layers)
            ]
        )
        self.fc = torch.nn.Linear(filter_size, 1, bias=True)

    @property
    def input_types(self):
        return {
            "enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "enc_mask": NeuralType(('B', 'T', 1), TokenDurationType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T'), EncodedRepresentation()),
        }

    def forward(self, enc, enc_mask):
        enc = torch.detach(enc)
        out = enc * enc_mask
        # out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.layers(out).transpose(1, 2)
        out = self.fc(out) 
        out = out * enc_mask.transpose(1, 2)
        return out.squeeze(-1)

def average_features(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg