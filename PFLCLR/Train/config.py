import os
from sacred import Experiment
ex = Experiment("XrayCLR")
base_dir = os.path.dirname(__file__)

@ex.config
def config():
    # basic
    name = "XrayCLR" # <clip/init>_<BCE/FOCAL>_<bs>
    seed = 2023
    device = 'cuda' # cuda
    version = None

    # logging
    log_dir = os.path.abspath(os.path.join(base_dir, "logs"))
    wandb_log = False
    wandb_api_key_fp = '/notebooks/R2Gen_Clip/wandb_api_key.txt'

    # dataset
    dataset_name = 'iu_xray' # or 'mimic_cxr'
    image_dir_iu = 'data/iu_xray/images/'
    image_dir_mimic = '/storage/mimic/mimic1/files/ '
    ann_path_iu = 'data/iu_xray/annotation.json'
    ann_path_mimic = '/notebooks/mimic/annotation.json'

    # Data loader settings
    data_workers = 12

    # training
    loss="infoNCE" # {"infoNCE", "BCE"}
    batch_size = 16
    lr = 0.00015 # 0.00004
    max_epochs = 100
    optimizer = "adamw" # adam or adamw
    decay_power = "no_decay" # no_decay, poly, cosine, linear
    warmup_steps = 6000 # https://chat.openai.com/share/ff341d8f-77dc-4a57-bc3b-a47210fe6b2e
    end_lr = 0.0 # for poly decay
    poly_decay_power = 1 # for poly decay
    ckpt_path = None # for resume training
    functional_test_size = None

    # model
    model_dir = os.path.abspath(os.path.join(base_dir, "ckpts"))

    # visual_extractor
    visual_extractor = 'resnet101' # {'resnet101', 'clip', 'medclip'}
    clip_path = '/notebooks/R2Gen_Clip/pretrain_weights/clip-imp-pretrained_128_6_after_4.pt'
    medclip_path = '/notebooks/R2Gen_Clip/pretrain_weights/medclip-vit/pytorch_model.bin'

    # clr_type = 'angle_pair' # {'angle_pair', 'persample'}