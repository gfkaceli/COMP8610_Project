import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

# Get available model names from torchvision models
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
# Update default architecture to resnet50 as per help text
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
# Updated default number of workers to match help text
parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
                    help='number of data loading workers (default: 18)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--disable-cuda', action='store_true',
                    help='disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='use 16-bit precision GPU training')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='number of views for contrastive learning training')
parser.add_argument('--gpu-index', default=0, type=int,
                    help='GPU index to use')


def main():
    args = parser.parse_args()
    # Ensure only two view training is supported.
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # Set device based on CUDA availability
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Initialize seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load and prepare dataset
    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=(args.workers!=0),
        pin_memory=False,
        drop_last=True
    )

    # Initialize model and move it to the selected device
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    model.to(args.device)
    print(args.device.type)
    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    # Run training using the specified GPU device (if any)
    if args.device.type == 'cuda':
        with torch.cuda.device(args.gpu_index):
            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            simclr.train(train_loader)
    else:
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
