"""
ResNet and Skip Connections - Deep Learning Solution

This solution demonstrates:
1. ResNet architectures (ResNet-18, ResNet-34, ResNet-50, ResNet-101)
2. Skip connections and residual learning
3. Training from scratch comparison
4. Bottleneck vs basic blocks
5. Learning curves and convergence analysis
6. Ablation studies on skip connections
7. Feature map visualization
8. Gradient flow analysis

Dataset: CIFAR-10/CIFAR-100 for image classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Dict
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class BasicBlock(nn.Module):
    """Basic ResNet block with 2 3x3 convolutions"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block with 1x1, 3x3, 1x1 convolutions"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet architecture with configurable depth"""

    def __init__(self, block, layers, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class PlainNet(nn.Module):
    """Plain network without skip connections for comparison"""

    def __init__(self, num_classes=10):
        super(PlainNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Plain convolutional layers (no skip connections)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=10):
    """ResNet-18 architecture"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=10):
    """ResNet-34 architecture"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=10):
    """ResNet-50 architecture"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=10):
    """ResNet-101 architecture"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


class GradientAnalyzer:
    """Analyze gradient flow in the network"""

    def __init__(self):
        self.gradient_norms = defaultdict(list)

    def register_hooks(self, model):
        """Register hooks to capture gradients"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self._save_gradient(name, grad))

    def _save_gradient(self, name, grad):
        """Save gradient norm"""
        if grad is not None:
            self.gradient_norms[name].append(grad.norm().item())

    def get_statistics(self):
        """Get gradient statistics"""
        stats = {}
        for name, norms in self.gradient_norms.items():
            stats[name] = {
                'mean': np.mean(norms),
                'std': np.std(norms),
                'max': np.max(norms),
                'min': np.min(norms)
            }
        return stats


def prepare_data(batch_size=128, dataset='cifar10'):
    """Prepare CIFAR-10 or CIFAR-100 dataset"""

    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Test transformation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        num_classes = 10
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        num_classes = 100

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    return trainloader, testloader, num_classes


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc, all_preds, all_targets


def train_model(model, trainloader, testloader, epochs=100, lr=0.1, name='model'):
    """Train model with learning rate scheduling"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    best_acc = 0

    print(f"\nTraining {name}...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, testloader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

        epoch_time = time.time() - start_time

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
                  f"Time: {epoch_time:.2f}s")

    print(f"Best Test Accuracy: {best_acc:.2f}%")

    return history, best_acc


def plot_learning_curves(histories, names):
    """Plot learning curves for multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training loss
    for history, name in zip(histories, names):
        axes[0, 0].plot(history['train_loss'], label=name)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Test loss
    for history, name in zip(histories, names):
        axes[0, 1].plot(history['test_loss'], label=name)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training accuracy
    for history, name in zip(histories, names):
        axes[1, 0].plot(history['train_acc'], label=name)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Test accuracy
    for history, name in zip(histories, names):
        axes[1, 1].plot(history['test_acc'], label=name)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resnet_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def ablation_study(trainloader, testloader, num_classes, epochs=50):
    """Ablation study on skip connections"""
    print("\n" + "="*80)
    print("ABLATION STUDY: Skip Connections Impact")
    print("="*80)

    # Plain network (no skip connections)
    plain_net = PlainNet(num_classes=num_classes).to(device)
    history_plain, acc_plain = train_model(plain_net, trainloader, testloader,
                                          epochs=epochs, name='PlainNet (No Skip)')

    # ResNet-18 (with skip connections)
    resnet = resnet18(num_classes=num_classes).to(device)
    history_resnet, acc_resnet = train_model(resnet, trainloader, testloader,
                                            epochs=epochs, name='ResNet-18 (Skip)')

    # Compare results
    print("\n" + "-"*80)
    print("COMPARISON RESULTS:")
    print(f"PlainNet (No Skip): {acc_plain:.2f}%")
    print(f"ResNet-18 (Skip):   {acc_resnet:.2f}%")
    print(f"Improvement:        {acc_resnet - acc_plain:.2f}%")
    print("-"*80)

    # Plot comparison
    plot_learning_curves([history_plain, history_resnet],
                        ['PlainNet (No Skip)', 'ResNet-18 (Skip)'])

    return {'plain': history_plain, 'resnet': history_resnet}


def compare_architectures(trainloader, testloader, num_classes, epochs=50):
    """Compare different ResNet architectures"""
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON")
    print("="*80)

    architectures = [
        (resnet18(num_classes=num_classes), 'ResNet-18'),
        (resnet34(num_classes=num_classes), 'ResNet-34'),
    ]

    histories = []
    names = []
    accuracies = []

    for model, name in architectures:
        model = model.to(device)
        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, name=name)
        histories.append(history)
        names.append(name)
        accuracies.append(acc)

    # Plot comparison
    plot_learning_curves(histories, names)

    # Summary
    print("\n" + "-"*80)
    print("ARCHITECTURE COMPARISON RESULTS:")
    for name, acc in zip(names, accuracies):
        print(f"{name:15s}: {acc:.2f}%")
    print("-"*80)

    return histories, names, accuracies


def visualize_feature_maps(model, dataloader, num_samples=5):
    """Visualize feature maps from different layers"""
    model.eval()

    # Get a batch of images
    inputs, _ = next(iter(dataloader))
    inputs = inputs[:num_samples].to(device)

    # Hook to capture feature maps
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))

    # Forward pass
    with torch.no_grad():
        _ = model(inputs)

    # Visualize
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))

    for i in range(num_samples):
        # Original image
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        # Feature maps from each layer
        for j, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            feature_map = activations[layer_name][i, 0].cpu().numpy()
            axes[i, j+1].imshow(feature_map, cmap='viridis')
            axes[i, j+1].set_title(f'{layer_name}')
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.savefig('resnet_feature_maps.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    print("="*80)
    print("ResNet and Skip Connections - Comprehensive Analysis")
    print("="*80)

    # Prepare data
    print("\nPreparing CIFAR-10 dataset...")
    trainloader, testloader, num_classes = prepare_data(batch_size=128, dataset='cifar10')

    # 1. Ablation Study: Impact of Skip Connections
    ablation_results = ablation_study(trainloader, testloader, num_classes, epochs=50)

    # 2. Architecture Comparison
    histories, names, accuracies = compare_architectures(trainloader, testloader,
                                                         num_classes, epochs=50)

    # 3. Visualize Feature Maps
    print("\nVisualizing feature maps...")
    model = resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('resnet18.pth', map_location=device)
                         if torch.cuda.is_available() else {})
    visualize_feature_maps(model, testloader)

    # 4. Gradient Flow Analysis
    print("\nAnalyzing gradient flow...")
    model = resnet18(num_classes=num_classes).to(device)
    analyzer = GradientAnalyzer()
    analyzer.register_hooks(model)

    # Train for a few steps to collect gradients
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for i, (inputs, targets) in enumerate(trainloader):
        if i >= 10:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    gradient_stats = analyzer.get_statistics()

    # Print gradient statistics
    print("\nGradient Flow Statistics:")
    print("-" * 80)
    for name, stats in list(gradient_stats.items())[:10]:
        print(f"{name:40s} Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Generated visualizations:")
    print("  - resnet_learning_curves.png")
    print("  - resnet_feature_maps.png")


if __name__ == "__main__":
    main()
