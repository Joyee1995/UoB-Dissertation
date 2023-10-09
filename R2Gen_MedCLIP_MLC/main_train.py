import os
import wandb
import torch
import argparse
import numpy as np
from argparser import parse_agrs
from modules.tokenizers import Tokenizer
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel

def main():

    # parse arguments
    args = parse_agrs()
    args.save_dir = os.path.join(args.save_dir, args.wandb_run_name)

    if args.wandb_log:
        with open(args.wandb_api_key_fp, 'r') as f:
            wandb_api_key = f.read().strip()
            wandb.login(key=wandb_api_key)

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume='allow',
            config=vars(args)
        )
    
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # build model architecture
    model = R2GenModel(args, tokenizer)
    if args.pretrained_r2gen is not None:
        model.load_state_dict(torch.load(args.pretrained_r2gen)['state_dict'])
    if args.xrayclr=="":
        if args.radgraph:
            model.build_mlc_layers_radgraph()
        else:
            model.build_mlc_layers()
    else:
        model.build_mlc_layers_combine()

    if args.use_clip and args.use_medclip:
        assert False, "can't use both clip and medclip"
    if args.use_clip:
        print("clip is used")
        model.load_clip_as_visual_extractor(args)
        
    if args.use_medclip and args.xrayclr=="":
        print("medclip is used")
        model.load_medclip_as_visual_extractor(args)
    elif args.use_medclip and args.xrayclr!="":
        print("medclip xrayclr combine is used")
        model.load_medclip_xrayclr_as_visual_extractor_combine(args)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    main()
