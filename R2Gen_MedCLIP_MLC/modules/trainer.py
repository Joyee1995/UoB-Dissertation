import os
from abc import abstractmethod

import time
import wandb
import torch
import pandas as pd
from tqdm import tqdm
from numpy import inf
from .loss import BCELoss
from modules.dataloaders import R2DataLoader


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args,lr_scheduler):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.wandb_log = args.wandb_log

        

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)
            if self.wandb_log:  
                wandb.log(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, log['test_BLEU_4'], save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, bleu4, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'scheduler': self.lr_scheduler.state_dict()
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if bleu4 > 0.10:
            best_path = os.path.join(self.checkpoint_dir, f'model_gt010_{epoch}.pth')
            torch.save(state, best_path)
            print("Saving current weights gt010 ...")

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args,lr_scheduler)
        self.dataset_name = args.dataset_name
        # self.lr_scheduler = lr_scheduler
        self.train_dataloader = R2DataLoader(args, self.model.tokenizer, split='train')
        self.val_dataloader = R2DataLoader(args, self.model.tokenizer, split='val', shuffle=False)
        self.test_dataloader = R2DataLoader(args, self.model.tokenizer, split='test', shuffle=False)
        self.loss_mlc = BCELoss()
        self.loss_cap_weight = args.loss_cap_weight
        self.loss_mlc_weight = args.loss_mlc_weight

        self.mlc = args.mlc

    def calculate_mlc_loss_acc(self, labels_mlc, output_mlc):
        if self.dataset_name == 'iu_xray':
            labels0_mlc, labels1_mlc = labels_mlc[:, 0], labels_mlc[:, 1]
            output0_mlc, output1_mlc = output_mlc 

            loss0_mlc = self.loss_mlc(output0_mlc, labels0_mlc)
            loss1_mlc = self.loss_mlc(output1_mlc, labels1_mlc)
            loss_mlc = (loss0_mlc + loss1_mlc) / 2

            acc0 = torch.mean(((torch.sigmoid(output0_mlc) > 0.5) == (labels0_mlc > 0.5)).type(torch.float32))
            acc1 = torch.mean(((torch.sigmoid(output1_mlc) > 0.5) == (labels1_mlc > 0.5)).type(torch.float32))
            acc = (acc0 + acc1) / 2
        else:
            loss_mlc = self.loss_mlc(output_mlc, labels_mlc)
            acc = torch.mean(((torch.sigmoid(output_mlc) > 0.5) == (labels_mlc > 0.5)).type(torch.float32))
        return loss_mlc, acc
    
    def _train_epoch(self, epoch):
        self.train_dataloader = R2DataLoader(self.args, self.model.tokenizer, split='train')

        print(f"training epoch {epoch}: ")
        self.model.train()
        train_loss, train_loss_cap, train_loss_mlc, train_acc = 0, 0, 0, 0
        for batch_idx, (batch) in enumerate(tqdm(self.train_dataloader)):
            images_id, images, reports_ids, reports_masks, labels_mlc = batch # labels_mlc.shape = B, 2, 5
            images, reports_ids, reports_masks, labels_mlc = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device), labels_mlc.to(self.device)
            
            if self.mlc:
                # print("trainer  mlc")
                output, output_mlc = self.model(images, reports_ids, mode='train') # output_mlc.shape = B, 5
            else:
                # print("trainer no mlc")
                output= self.model(images, reports_ids, mode='train')

            
            if self.mlc:
                loss_cap = self.criterion(output, reports_ids, reports_masks)
                loss_mlc, acc = self.calculate_mlc_loss_acc(labels_mlc, output_mlc)
                loss = self.loss_cap_weight * loss_cap + self.loss_mlc_weight * loss_mlc
                train_loss_cap += loss_cap.item()
                train_loss_mlc += loss_mlc.item()
                train_acc += acc.item()
            else:
                loss = self.criterion(output, reports_ids, reports_masks)

            train_loss += loss.item()
            

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        if self.mlc:
            log = {
                'train_loss_cap': train_loss_cap / len(self.train_dataloader),
                'train_loss_mlc': train_loss_mlc / len(self.train_dataloader),
                'train_loss': train_loss / len(self.train_dataloader),
                'train_acc': train_acc / len(self.train_dataloader)
                }
        else:
            log = {
                'train_loss': train_loss / len(self.train_dataloader),
                }


        print(f"validatin epoch {epoch}: ")
        self.model.eval()
        val_loss, val_loss_cap, val_loss_mlc, val_acc = 0, 0, 0, 0
        with torch.no_grad():
            val_acc = 0
            val_gts, val_res = [], []
            for batch_idx, (batch) in enumerate(tqdm(self.val_dataloader)):
                images_id, images, reports_ids, reports_masks, labels_mlc = batch
                images, reports_ids, reports_masks, labels_mlc = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device), labels_mlc.to(self.device)
                if self.mlc:
                    output, output_mlc = self.model(images, mode='sample')
                else:
                    output = self.model(images, mode='sample')

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                if self.mlc:
                    loss_mlc, acc = self.calculate_mlc_loss_acc(labels_mlc, output_mlc)
                    val_acc += acc.item()

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            if self.mlc:
                log.update({'val_acc': val_acc / len(self.val_dataloader)})


        print(f"testing epoch {epoch}: ")
        self.model.eval()
        test_loss, test_loss_cap, test_loss_mlc, test_acc = 0, 0, 0, 0
        with torch.no_grad():
            test_acc = 0
            test_gts, test_res = [], []
            for batch_idx, (batch) in enumerate(tqdm(self.test_dataloader)):
                images_id, images, reports_ids, reports_masks, labels_mlc = batch
                images, reports_ids, reports_masks, labels_mlc = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device), labels_mlc.to(self.device)
                if self.mlc:
                    output, output_mlc = self.model(images, mode='sample')
                else:
                    output = self.model(images, mode='sample')


                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                if self.mlc:
                    loss_mlc, acc = self.calculate_mlc_loss_acc(labels_mlc, output_mlc)
                    test_acc += acc.item()

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            if self.mlc:
                log.update({'test_acc': test_acc / len(self.test_dataloader)})

        self.lr_scheduler.step()

        return log
