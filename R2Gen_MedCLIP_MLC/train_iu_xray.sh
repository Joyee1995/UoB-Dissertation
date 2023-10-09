python3 main_train.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--step_size 50 \
--early_stop 100 \
--gamma 0.1 \
--wandb_log \
--wandb_project R2GenMLCV2 \
--wandb_run_id medclip_mlc_n_lc050_lm050 \
--wandb_run_name medclip_mlc_n_lc050_lm050 \
--wandb_api_key_fp /notebooks/R2Gen_Clip/wandb_api_key.txt \
--use_medclip \
--mlc \
--medclip_path /notebooks/R2Gen_Clip/pretrain_weights/clip-imp-pretrained_128_6_after_4.pt \
--loss_cap_weight 0.50 \
--loss_mlc_weight 0.50 \
--seed 9223

# --resume /notebooks/R2Gen_Clip/resultsV2/iu_xray/medclip_mlc_n_lc040_lm060/current_checkpoint.pth \



