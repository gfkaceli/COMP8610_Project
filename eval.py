import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from models.resnet_simclr import ResNetSimCLR


# A simple linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Pass images through the SimCLR model to obtain features.
            # Note: In many evaluation protocols, one uses the feature representation
            # before the projection head. If needed, modify the model to output those.
            feats = model(images)
            features_list.append(feats.cpu())
            labels_list.append(labels)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels


def train_linear_classifier(classifier, train_features, train_labels, device, epochs=50, lr=0.01):
    classifier.train()
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(train_features, train_labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for feats, labels in dataloader:
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def evaluate_linear_classifier(classifier, test_features, test_labels, device):
    classifier.eval()
    with torch.no_grad():
        test_features = test_features.to(device)
        outputs = classifier(test_features)
        preds = outputs.argmax(dim=1)
        accuracy = (preds.cpu() == test_labels).float().mean().item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SimCLR Representation via Linear Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the SimCLR checkpoint')
    parser.add_argument('--data', type=str, default='./datasets', help='Path to the dataset folder')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10', help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for feature extraction')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes (10 for CIFAR-10, 10 for STL10 unlabeled evaluation may differ)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs for linear classifier')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for linear classifier training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a basic transform for evaluation (deterministic and without augmentation)
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_dataset = datasets.CIFAR10(args.data, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(args.data, train=False, transform=transform, download=True)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # For STL10, using the labeled split for evaluation.
        train_dataset = datasets.STL10(args.data, split='train', transform=transform, download=True)
        test_dataset = datasets.STL10(args.data, split='test', transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load the SimCLR model checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch = checkpoint.get('arch', 'resnet50')
    # Determine out_dim by looking at one of the parameters, e.g., the final layer weight shape
    out_dim = checkpoint['state_dict']['backbone.fc.2.weight'].size(0)
    model = ResNetSimCLR(base_model=arch, out_dim=out_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # Freeze the SimCLR model's parameters
    for param in model.parameters():
        param.requires_grad = False

    # Extract features from the train and test sets
    print("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)

    # The feature dimension is determined by the extracted features
    feature_dim = train_features.size(1)
    classifier = LinearClassifier(feature_dim, args.num_classes).to(device)

    print("Training linear classifier...")
    train_linear_classifier(classifier, train_features, train_labels, device, epochs=args.epochs, lr=args.lr)
    print("Evaluating classifier...")
    evaluate_linear_classifier(classifier, test_features, test_labels, device)


if __name__ == '__main__':
    main()