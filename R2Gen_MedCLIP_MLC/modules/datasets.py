import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.functional_test_size = args.functional_test_size
        self.radgraph = args.radgraph

        self.examples = self.ann[self.split][:]
        if self.functional_test_size != -1:
            self.examples = self.examples[:self.functional_test_size]
            
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

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
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        label_1 = torch.load(os.path.join(self.image_dir, image_path[0]).replace(".png", ".pt"), map_location=torch.device('cpu'))
        label_2 = torch.load(os.path.join(self.image_dir, image_path[1]).replace(".png", ".pt"), map_location=torch.device('cpu'))
        label_1.requires_grad = False
        label_2.requires_grad = False
        labels_mlc = torch.stack([label_1, label_2]) # 2, 5
        sample = (image_id, image, report_ids, report_masks, seq_length, labels_mlc)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        # image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image = Image.open(os.path.join(example['image_dir'], image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        if  self.radgraph:
            labels_mlc = torch.load(os.path.join(example['image_dir'], image_path).replace(".jpg", "_radgraph.pt"))
        else:
            labels_mlc = torch.load(os.path.join(example['image_dir'], image_path).replace(".jpg", ".pt"))
        labels_mlc.requires_grad = False
        sample = (image_id, image, report_ids, report_masks, seq_length, labels_mlc)
        return sample


class IuxrayMultiImageDatasetLabel(BaseDataset):
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
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, image_path)
        return sample

# use the given labels
# class MimiccxrSingleImageDatasetLabel(BaseDataset): # TODO not complete -> fix image path
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_id = example['id']
#         image_path = example['image_path']
#         image = Image.open(os.path.join(example['image_dir'], image_path)).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#         report_ids = example['ids']
#         report_masks = example['mask']
#         seq_length = len(report_ids)
#         sample = (image_id, image, report_ids, report_masks, seq_length, image_path)
#         return sample
