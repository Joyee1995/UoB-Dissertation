import os
import json
import torch
import numpy as np
import torchmetrics
import torch.nn as nn
from Loss import get_loss_func
import pytorch_lightning as pl
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
from open_clip import _build_vision_tower, VisionTransformer
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()
        model = getattr(models, 'resnet101')(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

class VisionTransformerMed(nn.Module):
    def __init__(self, clip_path, embed_dim, **kwargs):
        super().__init__()
        clip_config = {
            "embed_dim": 512,
            "vision_cfg": {
                "image_size": 224,
                "layers": 12,
                "width": 768,
                "patch_size": 32,
                "output_tokens": True

            },
        }
        self.visual = _build_vision_tower(clip_config['embed_dim'], clip_config['vision_cfg'])
        state_dict = torch.load(clip_path, map_location="cpu")
        if type(state_dict) is torch.jit._script.RecursiveScriptModule:
            state_dict = state_dict.state_dict()
        state_dict = {name.replace("visual.", ""): weights for name, weights in state_dict.items() if "visual" in name}
        self.visual.load_state_dict(state_dict)
        self.proj = nn.Parameter(torch.randn(clip_config['vision_cfg']['width'], embed_dim))

    def forward(self, x):
        pooled, tokens = self.visual(x)
        pooled = pooled @ self.proj
        tokens = torch.bmm(tokens, self.proj.repeat(tokens.shape[0], 1, 1))
        return tokens, pooled

class SwinTransformerMed(nn.Module):
    def __init__(self, medclip_path, embed_dim):
        super().__init__()
        self.model = AutoModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        self.proj = nn.Parameter(torch.randn(768, embed_dim))
        self.load_pretrained(medclip_path)

    def load_pretrained(self, medclip_path):
        """
        medclip_path: "/notebooks/MedClip/pretrained/medclip-vit/pytorch_model.bin"
        """
        state_dict = torch.load(medclip_path)
        parent_attribute = "vision_model.model."
        for p, w in state_dict.copy().items():
            if parent_attribute in p:
                state_dict[p.replace(parent_attribute, "")] = state_dict[p]
            del state_dict[p]
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        output = self.model(x)
        pooled, tokens = output['pooler_output'], output['last_hidden_state']
        pooled = pooled @ self.proj
        tokens = torch.bmm(tokens, self.proj.repeat(tokens.shape[0], 1, 1))
        return tokens, pooled

class XrayCLR(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.lr = _config["lr"]
        self.optimizer = _config["optimizer"]

        self.decay_power = _config['decay_power']
        self.warmup_steps = _config['warmup_steps']
        self.max_steps = _config['max_steps']
        self.end_lr = _config['end_lr']
        self.poly_decay_power = _config['poly_decay_power']
        self.num_classes = _config['batch_size']

        if _config['visual_extractor'] == 'resnet101':
            self.visual_extractor = ResNet101()
        elif _config['visual_extractor'] == 'clip':
            self.visual_extractor = VisionTransformerMed(_config['clip_path'], 2048)
        elif _config['visual_extractor'] == 'medclip':
            self.visual_extractor = SwinTransformerMed(_config['medclip_path'], 2048)
        else:
            NotImplementedError(f"Unknown visual_extractor: {_config['visual_extractor']}")
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        metric_collection = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes),
            torchmetrics.classification.MulticlassAveragePrecision(num_classes=self.num_classes),
            torchmetrics.classification.MulticlassPrecision(num_classes=self.num_classes),
            torchmetrics.classification.MulticlassRecall(num_classes=self.num_classes)
        ])
        self.train_metrics = metric_collection.clone(prefix='train_')
        self.valid_metrics = metric_collection.clone(prefix='valid_')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
    def set_loss_func(self, loss_name): # InfoNCE recommend
        self.loss_func = get_loss_func(loss_name)

    def load_ckpt_state_dict(self, ckpt_fp):
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)
    
    def forward_single(self, image_tensor):
        att_feats_0, fc_feats_0 = self.visual_extractor(image_tensor)
        frame_feat = fc_feats_0 / fc_feats_0.norm(dim=1, keepdim=True)
        return frame_feat

    def forward(self, batch, targets=None):
        images_id, images, labels_mlc = batch # labels_mlc.shape = B, 2, 5
        fc_feats_0 = self.forward_single(images[:, 0])
        fc_feats_1 = self.forward_single(images[:, 1])
        return fc_feats_0, fc_feats_1
    
    def training_step(self, batch, batch_idx):
        fc_feats_0, fc_feats_1 = self(batch)
        image_logits = fc_feats_0 @ fc_feats_1.t() * self.logit_scale.exp()        
        ground_truth = torch.eye(image_logits.shape[0]).to(image_logits.device)
        loss = self.loss_func(image_logits, ground_truth)
        if image_logits.shape[0] < self.num_classes:
            pad_size = self.num_classes - image_logits.shape[0]
            image_logits = torch.nn.functional.pad(image_logits, (0, pad_size, 0, pad_size), mode='constant', value=0)
            ground_truth = torch.nn.functional.pad(ground_truth, (0, pad_size, 0, pad_size), mode='constant', value=0)
        self.train_metrics.update(torch.softmax(image_logits, dim=1), torch.argmax(ground_truth, dim=1))
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        _train_metrics = self.train_metrics.compute()
        self.log_dict(_train_metrics)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        fc_feats_0, fc_feats_1 = self(batch)
        image_logits = fc_feats_0 @ fc_feats_1.t() * self.logit_scale.exp()        
        ground_truth = torch.eye(image_logits.shape[0]).to(image_logits.device)
        loss = self.loss_func(image_logits, ground_truth)
        if image_logits.shape[0] < self.num_classes:
            pad_size = self.num_classes - image_logits.shape[0]
            image_logits = torch.nn.functional.pad(image_logits, (0, pad_size, 0, pad_size), mode='constant', value=0)
            ground_truth = torch.nn.functional.pad(ground_truth, (0, pad_size, 0, pad_size), mode='constant', value=0)
        self.valid_metrics.update(torch.softmax(image_logits, dim=1), torch.argmax(ground_truth, dim=1))
        self.log("valid_loss", loss)

    def on_validation_epoch_end(self):
        _valid_metrics = self.valid_metrics.compute()
        self.log_dict(_valid_metrics)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-6, betas=(0.9, 0.98))
        else:
            assert False, f"Unknown optimizer: {optimizer}"

        if self.decay_power == "no_decay":
            return optimizer
        else:
            if self.decay_power == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=self.warmup_steps, 
                    num_training_steps=self.max_steps
                )
            elif self.decay_power == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.max_steps,
                )
            elif self.decay_power == "poly":
                scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=self.end_lr,
                    power=self.poly_decay_power,
                )
            sched = {"scheduler": scheduler, "interval": "step"}

            return ([optimizer], [sched])    
