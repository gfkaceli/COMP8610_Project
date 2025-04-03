import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import pytorch_lightning as pl

from models.resnet_simclr import ResNetSimCLR  # Your SimCLR model

# Define the logistic regression classifier as a PyTorch Lightning Module
class LogisticRegression(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.hparams.max_epochs * 0.6), int(self.hparams.max_epochs * 0.8)],
            gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(mode + '_loss', loss, prog_bar=True)
        self.log(mode + '_acc', acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')


def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images)
            features_list.append(feats.cpu())
            labels_list.append(labels)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SimCLR representations using Logistic Regression with PyTorch Lightning")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the SimCLR checkpoint')
    parser.add_argument('--data', type=str, default='./datasets', help='Path to dataset folder')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10',
                        help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for feature extraction')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for logistic regression')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for logistic regression')
    parser.add_argument('--max-epochs', type=int, default=100, help='Max epochs for logistic regression training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define evaluation transforms (no augmentation, only normalization)
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.247, 0.243, 0.261])
        ])
        train_dataset = datasets.CIFAR10(args.data, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(args.data, train=False, transform=transform, download=True)
        num_classes = 10
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        # For STL10, use the labeled training split
        train_dataset = datasets.STL10(args.data, split='train', transform=transform, download=True)
        test_dataset = datasets.STL10(args.data, split='test', transform=transform, download=True)
        num_classes = 10

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load the SimCLR checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch = checkpoint.get('arch', 'resnet50')
    # Determine out_dim by checking the saved state dict. Here we assume the final fc layer weight's first dimension.
    out_dim = checkpoint['state_dict']['backbone.fc.2.weight'].size(0)
    simclr_model = ResNetSimCLR(base_model=arch, out_dim=out_dim)
    simclr_model.load_state_dict(checkpoint['state_dict'])
    simclr_model = simclr_model.to(device)
    simclr_model.eval()
    # Freeze parameters
    for param in simclr_model.parameters():
        param.requires_grad = False

    print("Extracting training features...")
    train_features, train_labels = extract_features(simclr_model, train_loader, device)
    print("Extracting test features...")
    test_features, test_labels = extract_features(simclr_model, test_loader, device)

    # Create TensorDatasets for logistic regression training
    train_data = TensorDataset(train_features, train_labels)
    test_data = TensorDataset(test_features, test_labels)
    train_loader_lr = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader_lr = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Instantiate the logistic regression model
    feature_dim = train_features.size(1)
    lr_model = LogisticRegression(feature_dim=feature_dim,
                                  num_classes=num_classes,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  max_epochs=args.max_epochs)

    # Train the logistic regression classifier using PyTorch Lightning
    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="auto")
    trainer.fit(lr_model, train_loader_lr, val_dataloaders=test_loader_lr)
    trainer.test(lr_model, test_loader_lr)

if __name__ == '__main__':
    main()