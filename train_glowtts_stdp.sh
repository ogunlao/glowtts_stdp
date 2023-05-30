
python glow_tts_with_pitch.py \
    --config-path="conf" \
    --config-name="glow_tts_stdp" \
    train_dataset="dataset/train_tts_common_voice.json" \
    validation_datasets="dataset/dev_tts_common_voice.json" \
    speaker_emb_path="../tts_experiments/embeddings/spk_emb_exp_1plus.pkl" \
    sup_data_path="../cv-corpus-7.0-2021-07-21/en/cv_mos4_all_sup_data_folder"
