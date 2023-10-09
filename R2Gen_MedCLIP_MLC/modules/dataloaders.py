import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, IuxrayMultiImageDatasetLabel

class CustomSampler(RandomSampler):
    def __init__(self, data_source, limit_train_batches):
        super().__init__(data_source)
        self.limit_train_batches = limit_train_batches

    def __iter__(self):
        num_samples = int(self.limit_train_batches * len(self.data_source))
        indices = torch.randperm(len(self.data_source), generator=self.generator).tolist()
        return iter(indices[:num_samples])

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle=False):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.limit_train_batches = args.limit_train_batches

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
        }
        if (self.split == 'train') and (self.limit_train_batches != 1):
            self.init_kwargs['sampler'] = CustomSampler(self.dataset, self.limit_train_batches)

        super().__init__(**self.init_kwargs)

    def __len__(self):
        if (self.split == 'train') and (self.limit_train_batches != 1):
            return int(self.limit_train_batches * super().__len__())
        else:
            return super().__len__()

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths, labels_mlc = zip(*data)
        images = torch.stack(images, 0)
        labels_mlc = torch.stack(labels_mlc, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), labels_mlc # labels_mlc: B, 2, 5

class R2DataLoaderLabel(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDatasetLabel(self.args, self.tokenizer, self.split, transform=self.transform)
        # else:
        #     self.dataset = MimiccxrSingleImageDatasetLabel(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths, image_paths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), image_paths

