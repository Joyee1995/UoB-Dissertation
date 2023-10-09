import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
import numpy as np
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)


class BaseDataset(Dataset):
    def __init__(self, _config, split):
        self.image_dir_iu = _config['image_dir_iu']
        self.image_dir_mimic = _config['image_dir_mimic']
        self.ann_path_iu = _config['ann_path_iu']
        self.ann_path_mimic = _config['ann_path_mimic']
        self.split = split
        self.ann_iu = json.loads(open(self.ann_path_iu, 'r').read())
        self.ann_mimic = json.loads(open(self.ann_path_mimic, 'r').read())
        if self.split == 'train':
            for i in self.ann_iu[self.split]:
                i['id'] = i['id']+'_iu'
                # print(i)
            self.examples = self.ann_iu[self.split]+self.ann_mimic[self.split]
        else:
            for i in self.ann_iu['val']:
                i['id'] = i['id']+'_iu'
            for i in self.ann_iu['test']:
                i['id'] = i['id']+'_iu'
                # print(i)
            self.examples = self.ann_iu['val'] + self.ann_iu['test']+self.ann_mimic['val'] + self.ann_mimic['test']
        print(f"length of samples ({self.split}): ", len(self.examples))

        if _config['functional_test_size'] is not None:
            self.examples = self.examples[:_config['functional_test_size']]

        if self.split == 'train':
            # self.transform = Compose(
            #         [
            #             transforms.Resize(256),
            #             ScaleIntensity(),
            #             RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            #             RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            #             transforms.ToTensor(),
            #             transforms.Normalize((0.485, 0.456, 0.406),
            #                         (0.229, 0.224, 0.225))
            #         ]
            #     )

            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.GaussianBlur(5),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),     # Randomly rotate the image by up to 10 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness and contrast
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)

        label_1 = torch.load(os.path.join(self.image_dir, image_path[0]).replace(".png", ".pt"), map_location=torch.device('cpu'))
        label_2 = torch.load(os.path.join(self.image_dir, image_path[1]).replace(".png", ".pt"), map_location=torch.device('cpu'))
        label_1.requires_grad = False
        label_2.requires_grad = False
        labels_mlc = torch.stack([label_1, label_2]) # 2, 5
        sample = (image_id, image, labels_mlc)
        return sample


class MimiccxrMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(example['image_dir'], image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(example['image_dir'], image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)

        label_1 = torch.load(os.path.join(example['image_dir'], image_path[0]).replace(".jpg", ".pt"), map_location=torch.device('cpu'))
        label_2 = torch.load(os.path.join(example['image_dir'], image_path[1]).replace(".jpg", ".pt"), map_location=torch.device('cpu'))
        label_1.requires_grad = False
        label_2.requires_grad = False
        labels_mlc = torch.stack([label_1, label_2]) # 2, 5
        sample = (image_id, image, labels_mlc)
        return sample
    
class MimiccxrCombineIuMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        # print(image_id)
        if "_" in image_id and image_id.split("_")[-1]=="iu":
            image_path = example['image_path']
            image_1 = Image.open(os.path.join(self.image_dir_iu, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir_iu, image_path[1])).convert('RGB')
            if self.transform is not None:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
                if random.random() < 0.2: # 20% chance to apply horizontal flip
                    image_1 = transforms.functional.hflip(image_1)
                    image_2 = transforms.functional.hflip(image_2)
            image = torch.stack((image_1, image_2), 0)

            # label_1 = torch.load(os.path.join(self.image_dir_iu, image_path[0]).replace(".png", ".pt"), map_location=torch.device('cpu'))
            # label_2 = torch.load(os.path.join(self.image_dir_iu, image_path[1]).replace(".png", ".pt"), map_location=torch.device('cpu'))
            # label_1.requires_grad = False
            # label_2.requires_grad = False
            # labels_mlc = torch.stack([label_1, label_2]) # 2, 5
            labels_mlc = 0
            sample = (image_id, image, labels_mlc)
            return sample

        else:
            image_path = example['image_path']
            # print(example)
            image_1 = Image.open(os.path.join(example['image_dir'], image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(example['image_dir'], image_path[1])).convert('RGB')
            if self.transform is not None:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

            # label_1 = torch.load(os.path.join(example['image_dir'], image_path[0]).replace(".jpg", ".pt"), map_location=torch.device('cpu'))
            # label_2 = torch.load(os.path.join(example['image_dir'], image_path[1]).replace(".jpg", ".pt"), map_location=torch.device('cpu'))
            # label_1.requires_grad = False
            # label_2.requires_grad = False
            # labels_mlc = torch.stack([label_1, label_2]) # 2, 5
            labels_mlc = 0
            sample = (image_id, image, labels_mlc)
            return sample