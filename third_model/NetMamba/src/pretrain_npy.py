import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

# import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import (
    count_parameters, init_distributed_mode, get_rank, get_world_size,
    is_main_process, load_model
)

import models_net_mamba
from engine import pretrain_one_epoch
from contextlib import suppress
from typing import List, Optional, Tuple, Union
from torch import Tensor

def get_args_parser():
    parser = argparse.ArgumentParser('flow mamba pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--steps', default=150000, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_steps_freq', default=10000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='net_mamba_pretrain', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=25, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output/pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)

    # pretrain tasks
    parser.add_argument('--pop', action='store_true', help='packet order prediction')
    parser.add_argument('--pop_loss_weight', default=0.01, type=float,
                        help='packet order prediction loss weight')
    return parser


def normalize_array(tensor, mean, std, dtype=torch.float32):
    # Ensure mean and std are tensors and on the same device as the input tensor
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    # Check if std contains any zero values to prevent division by zero
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

    # Reshape mean and std to match the tensor shape if they are 1-dimensional
    if mean.ndim == 1:
        mean = mean.view(-1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1)
    # print('tensor_shape:', tensor.shape)
    # Perform normalization
    return tensor.sub_(mean).div_(std)

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor) / (255)


class NPYPipelineDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._make_dataset()
        self.label_to_idx = self._get_label_to_idx()


    def _make_dataset(self):
        samples = []
        for root, _, fnames in sorted(os.walk(self.root_dir)):
            # print('----------------------print_root-----------------------')
            # print(root)
            # print(fnames)
            for fname in sorted(fnames):
                if fname.endswith('.npy'):
                    path = os.path.join(root, fname)
                    label = os.path.basename(os.path.dirname(path))
                    samples.append((path, label))
        if not samples:
            raise RuntimeError(f"No .npy files found in {self.root_dir}")
        return samples

    def _get_label_to_idx(self):
        labels = sorted(set(label for _, label in self.samples))
        if not labels:
            raise RuntimeError("No labels found. Check your dataset structure.")
        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        # print('NPYPipelineDataset_data:', data)
        # print('NPYPipelineDataset_data_shape:', data.shape)
        data = min_max_normalize(data)

        if self.transform is not None:

            mean = [0.5]
            std = [0.5]
            data = normalize_array(data, mean, std)
            # data = self.transform(data)

        label_idx = self.label_to_idx[label]
        return data, label_idx



def main(args):
    init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    mean = 0.5
    std = 0.5



    # 数据转换和归一化
    # def normalize_array(array, mean=0.5, std=0.5):
    #     return (array - mean) / std

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def to_tensor(array):
        return torch.tensor(array, dtype=torch.float32)

    class ToTensor(object):
        def __call__(self, x):
            return torch.tensor(x, dtype=torch.float32)  # Add channel dimension

    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, x):
            return (x - self.mean) / self.std

    transform_train = transforms.Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5),
    ])

    # transform_train = transforms.Compose([
    #     # transforms.Lambda(lambda x: torch.nn.functional.normalize(to_tensor(x), mean, std)),
    #     transforms.Normalize(mean, std),
    #     transforms.Lambda(lambda x: to_tensor(x))
    # ])

    # transform_train = transforms.Compose([
    #     transforms.Lambda(lambda x: normalize_array(x, mean, std)),
    #     transforms.Lambda(lambda x: to_tensor(x))
    # ])
    # transform_train = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std),
    # ])

    dataset = NPYPipelineDataset(os.path.join(args.data_path), transform=transform_train)
    # print('-------------------------------------------NpyDataset!--------------------------')
    # print(len(dataset))

    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    print("Sampler_train = %s" % str(sampler))
    # print(len(sampler))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model

    model = models_net_mamba.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # print('-----------------------------------------------')
    # print(len(data_loader_train))
    epochs = int(args.steps / len(data_loader_train)) + 1
    args.epochs = epochs

    print(f"Start training for {args.steps} steps")
    start_time = time.time()
    for epoch in range(0, epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = pretrain_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            log_writer=log_writer,
            model_without_ddp=model_without_ddp,
            args=args
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }
        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    with open(os.path.join(args.output_dir, "train_stats.json"), mode="w", encoding="utf-8") as f:
        json.dump({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "total_time": total_time,
        }, f, indent=2)

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


