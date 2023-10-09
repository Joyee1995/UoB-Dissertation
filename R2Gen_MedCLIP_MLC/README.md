# R2Gen+MedCLIP

This work expands upon R2Gen. The repository structure follows the official R2Gen repository. For more details, visit [R2Gen repository](https://github.com/zhjohnchan/R2Gen).

## Requirements

For MedCLIP pretrained weight, download from the [MedCLIP repository](https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip).

For environment settings, refer to `r2gen_medclip.ipynb`.

For datasets, download from the [R2Gen repository](https://github.com/zhjohnchan/R2Gen) or alternatively, visit [MSAT repository](https://github.com/wang-zhanyu/MSAT/).

## Train

To train a model on the IU X-Ray data, execute: `bash train_iu_xray.sh`.

To train a model on the MIMIC-CXR data, execute: `bash train_mimic_cxr.sh`.

**Note:** Ensure that you check the image_dir and ann_path before running these commands.

## Test

To test a model on the IU X-Ray data, execute: `bash test_iu_xray.sh`.

To test a model on the MIMIC-CXR data, execute: `bash test_mimic_cxr.sh`.

After running the 'test' commands, a `pred.json` file will be generated. To further analyze the clinical efficacy metric, refer to `chexpert.ipynb` to generate pseudo-labels. Then, use `chexpert_score.ipynb` to calculate precision, recall, and the F1 score.


