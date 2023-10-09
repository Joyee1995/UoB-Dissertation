python3 main_train.py \
--image_dir /storage/mimic/mimic1/files/ \
--ann_path /notebooks/mimic/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 60 \
--threshold 3 \
--epochs 60 \
--step_size 50 \
--gamma 0.1 \
--wandb_log \
--wandb_project R2GenMIMIC \
--wandb_run_id medclip_mlc_n_lc050_lm050 \
--wandb_run_name medclip_mlc_n_lc050_lm050 \
--wandb_api_key_fp /notebooks/R2Gen_Clip/wandb_api_key.txt \
--save_dir results_mimic/mimic_cxr/medclip_mlc_n_lc050_lm050 \
--n_classes 14 \
--use_medclip \
--mlc \
--medclip_path /notebooks/R2Gen_Clip/pretrain_weights/medclip-vit/pytorch_model.bin \
--loss_cap_weight 0.50 \
--loss_mlc_weight 0.50 \
--batch_size 256 \
--seed 456789

# --resume /notebooks/R2Gen_Clip/results_mimic/mimic_cxr/medclip_mlc_n_lc050_lm050/medclip_mlc_n_lc050_lm050/current_checkpoint.pth \
