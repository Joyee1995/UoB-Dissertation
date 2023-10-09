import os
import json
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))
from transformers import AutoModel, AutoTokenizer
from open_clip import _build_vision_tower, VisionTransformer
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder

class VisionTransformerMed(nn.Module):
    def __init__(self, args, embed_dim, **kwargs):
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
        state_dict = torch.load(args.clip_path, map_location="cpu")
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
    def __init__(self, args, embed_dim):
        super().__init__()
        self.model = AutoModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        self.proj = nn.Parameter(torch.randn(768, embed_dim))
        self.load_pretrained(args.medclip_path)

    def load_pretrained(self, medclip_path):
        """
        medclip_path: "/notebooks/MedClip/pretrained/medclip-vit/pytorch_model.bin"
        """
        if "xrayclr" in os.path.basename(medclip_path).lower():
            state_dict = torch.load(medclip_path)['state_dict']
            parent_attribute = "visual_extractor.model."
            for p, w in state_dict.copy().items():
                if parent_attribute in p:
                    state_dict[p.replace(parent_attribute, "")] = state_dict[p]
                del state_dict[p]
        else:
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

class SwinTransformerMed_Combined(nn.Module):
    def __init__(self, args, embed_dim,path):
        super().__init__()
        self.model = AutoModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        self.proj = nn.Parameter(torch.randn(768, embed_dim))
        self.load_pretrained(path)

    def load_pretrained(self, medclip_path):
        """
        medclip_path: "/notebooks/MedClip/pretrained/medclip-vit/pytorch_model.bin"
        """
        if "xrayclr" in os.path.basename(medclip_path).lower():
            state_dict = torch.load(medclip_path)['state_dict']
            parent_attribute = "visual_extractor.model."
            for p, w in state_dict.copy().items():
                if parent_attribute in p:
                    state_dict[p.replace(parent_attribute, "")] = state_dict[p]
                del state_dict[p]
        else:
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

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.n_classes = args.n_classes
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.freeze_clip_new = args.freeze_clip_new
        self.freeze_medclip_new = args.freeze_medclip_new
        self.radgraph = args.radgraph
        
        if args.xrayclr!="":
            print("use combine forward")
            self.forward = self.forward_mimic_cxr_combine_xrayclr_medclip
        else:
            if args.mlc == True:
                print("with mlc")
                if args.dataset_name == 'iu_xray':
                    self.forward = self.forward_iu_xray
                else:
                    self.forward = self.forward_mimic_cxr
            else:
                print("no mlc")
                if args.dataset_name == 'iu_xray':
                    self.forward = self.forward_iu_xray_no_mlc
                else:
                    self.forward = self.forward_mimic_cxr_no_mlc
        
       

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
    def load_clip_as_visual_extractor(self, args):
        self.visual_extractor = VisionTransformerMed(args, 2048) 
        if self.freeze_clip_new:
            self.visual_extractor.requires_grad_(False)

    def load_medclip_as_visual_extractor(self, args):
        self.visual_extractor = SwinTransformerMed(args, 2048)
        if self.freeze_medclip_new:
            self.visual_extractor.requires_grad_(False)
    
    def load_medclip_xrayclr_as_visual_extractor_combine(self, args):
        self.visual_extractor1 = SwinTransformerMed_Combined(args, 2048, args.medclip_path)
        self.visual_extractor2 = SwinTransformerMed_Combined(args, 2048, args.xrayclr)
        if self.freeze_medclip_new:
            self.visual_extractor1.requires_grad_(False)
            self.visual_extractor2.requires_grad_(False)

    def build_mlc_layers(self):
        self.mlc_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, self.n_classes)
        )
    
    def build_mlc_layers_combine(self):
        self.mlc_layers_combine = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Linear(2048, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, self.n_classes)
        )
    
    def build_mlc_layers_radgraph(self):
        self.mlc_layers_radgraph = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, self.n_classes)
        )

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0]) # att_feats_0: B, 49, 2048, fc_feats_0.shape: B, 2048
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1) # B, 4096
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1) # att_feats_0: B, 98, 2048
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        
        B = fc_feats_0.shape[0]
        output_mlc = self.mlc_layers(torch.cat([fc_feats_0, fc_feats_1], dim=0)) # 2*B, 5
        return output, (output_mlc[:B], output_mlc[B:])

    def forward_iu_xray_no_mlc(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0]) # att_feats_0: B, 49, 2048, fc_feats_0.shape: B, 2048
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1) # B, 4096
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1) # att_feats_0: B, 98, 2048
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        
        # B = fc_feats_0.shape[0]
        # output_mlc = self.mlc_layers(torch.cat([fc_feats_0, fc_feats_1], dim=0)) # 2*B, 5
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        
        if self.radgraph:
            output_mlc = self.mlc_layers_radgraph(fc_feats)
        else:
            output_mlc = self.mlc_layers(fc_feats)
        return output, output_mlc

    def forward_mimic_cxr_combine_xrayclr_medclip(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor1(images)
        att_feats_1, fc_feats_1 = self.visual_extractor2(images)
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1) # B, 4096
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1) # att_feats_0: B, 98, 2048
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        output_mlc = self.mlc_layers_combine(torch.cat([fc_feats_0, fc_feats_1], dim=1)) # 2*B, 5
        return output, output_mlc

    def forward_mimic_cxr_no_mlc(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        
        # output_mlc = self.mlc_layers(fc_feats)
        return output

