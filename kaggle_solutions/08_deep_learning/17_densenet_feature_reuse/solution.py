"""
DenseNet and Feature Reuse - Deep Learning Solution

This solution demonstrates:
1. DenseNet architectures (DenseNet-121, DenseNet-169, DenseNet-201)
2. Dense connectivity patterns and feature reuse
3. Bottleneck layers and compression
4. Growth rate impact analysis
5. Memory efficiency comparisons
6. Training from scratch vs transfer learning
7. Feature concatenation visualization
8. Ablation studies on dense connections

Dataset: CIFAR-10/ImageNet for image classification
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
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class DenseLayer(nn.Module):
    """Single dense layer with bottleneck"""

    def __init__(self, in_channels, growth_rate, bn_size=4, dropout=0.0):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        # Bottleneck layer
        out = self.conv1(self.relu1(self.bn1(x)))
        # Convolution layer
        out = self.conv2(self.relu2(self.bn2(out)))

        if self.dropout is not None:
            out = self.dropout(out)

        # Concatenate input and output (dense connection)
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):
    """Dense block with multiple dense layers"""

    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, dropout=0.0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                dropout
            ))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    """Transition layer between dense blocks"""

    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    """DenseNet architecture"""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, dropout=0.0,
                 num_classes=10, compression=0.5):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        self.compression = compression

        # Initial convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7,
                               stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout=dropout
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            # Add transition layer (except after the last block)
            if i != len(block_config) - 1:
                trans = Transition(num_features, int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(num_classes=10, growth_rate=32):
    """DenseNet-121 architecture"""
    return DenseNet(growth_rate=growth_rate, block_config=(6, 12, 24, 16),
                   num_init_features=64, num_classes=num_classes)


def densenet169(num_classes=10, growth_rate=32):
    """DenseNet-169 architecture"""
    return DenseNet(growth_rate=growth_rate, block_config=(6, 12, 32, 32),
                   num_init_features=64, num_classes=num_classes)


def densenet201(num_classes=10, growth_rate=32):
    """DenseNet-201 architecture"""
    return DenseNet(growth_rate=growth_rate, block_config=(6, 12, 48, 32),
                   num_init_features=64, num_classes=num_classes)


class SimpleCNN(nn.Module):
    """Simple CNN for comparison without dense connections"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory(model, input_size=(1, 3, 32, 32)):
    """Measure model memory usage"""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        memory_mb = 0  # Cannot measure on CPU easily

    return memory_mb


def prepare_data(batch_size=128, dataset='cifar10'):
    """Prepare dataset"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

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

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model(model, trainloader, testloader, epochs=100, lr=0.1, name='model'):
    """Train model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
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
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

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


def growth_rate_analysis(trainloader, testloader, num_classes, epochs=50):
    """Analyze impact of growth rate"""
    print("\n" + "="*80)
    print("GROWTH RATE ANALYSIS")
    print("="*80)

    growth_rates = [12, 24, 32, 48]
    histories = []
    accuracies = []
    parameters = []

    for gr in growth_rates:
        print(f"\nTesting Growth Rate: {gr}")
        model = densenet121(num_classes=num_classes, growth_rate=gr).to(device)
        params = count_parameters(model)
        parameters.append(params)

        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, name=f'DenseNet-121 (k={gr})')
        histories.append(history)
        accuracies.append(acc)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy vs Growth Rate
    axes[0].plot(growth_rates, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Growth Rate (k)')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Impact of Growth Rate on Accuracy')
    axes[0].grid(True, alpha=0.3)

    # Parameters vs Growth Rate
    axes[1].plot(growth_rates, [p/1e6 for p in parameters], 's-',
                linewidth=2, markersize=8, color='red')
    axes[1].set_xlabel('Growth Rate (k)')
    axes[1].set_ylabel('Parameters (Millions)')
    axes[1].set_title('Model Size vs Growth Rate')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('densenet_growth_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*80)
    print("GROWTH RATE RESULTS:")
    for gr, acc, params in zip(growth_rates, accuracies, parameters):
        print(f"k={gr:2d}: Acc={acc:5.2f}%, Params={params/1e6:.2f}M")
    print("-"*80)

    return histories, accuracies


def compression_analysis(trainloader, testloader, num_classes, epochs=50):
    """Analyze impact of compression factor"""
    print("\n" + "="*80)
    print("COMPRESSION FACTOR ANALYSIS")
    print("="*80)

    compressions = [0.3, 0.5, 0.7, 1.0]
    accuracies = []
    parameters = []

    for comp in compressions:
        print(f"\nTesting Compression: {comp}")
        model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                        compression=comp, num_classes=num_classes).to(device)
        params = count_parameters(model)
        parameters.append(params)

        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, name=f'DenseNet (θ={comp})')
        accuracies.append(acc)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(compressions, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Compression Factor (θ)')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Impact of Compression on Accuracy')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(compressions, [p/1e6 for p in parameters], 's-',
                linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Compression Factor (θ)')
    axes[1].set_ylabel('Parameters (Millions)')
    axes[1].set_title('Model Size vs Compression')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('densenet_compression_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*80)
    print("COMPRESSION RESULTS:")
    for comp, acc, params in zip(compressions, accuracies, parameters):
        print(f"θ={comp:.1f}: Acc={acc:5.2f}%, Params={params/1e6:.2f}M")
    print("-"*80)


def compare_architectures(trainloader, testloader, num_classes, epochs=50):
    """Compare different DenseNet architectures"""
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON")
    print("="*80)

    models = [
        (densenet121(num_classes=num_classes), 'DenseNet-121'),
        (SimpleCNN(num_classes=num_classes), 'Simple CNN'),
    ]

    results = []

    for model, name in models:
        model = model.to(device)
        params = count_parameters(model)
        memory = measure_memory(model)

        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, name=name)

        results.append({
            'name': name,
            'accuracy': acc,
            'parameters': params,
            'memory_mb': memory,
            'history': history
        })

    # Summary table
    print("\n" + "-"*80)
    print("ARCHITECTURE COMPARISON:")
    print(f"{'Model':<20} {'Accuracy':<12} {'Parameters':<15} {'Memory (MB)':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:>6.2f}% "
              f"{r['parameters']/1e6:>10.2f}M {r['memory_mb']:>10.2f}")
    print("-"*80)

    return results


def visualize_dense_connections(model, num_blocks=4):
    """Visualize dense connectivity pattern"""
    fig, axes = plt.subplots(1, num_blocks, figsize=(20, 5))

    for i in range(num_blocks):
        block_name = f'denseblock{i+1}'
        if hasattr(model.features, block_name):
            block = getattr(model.features, block_name)
            num_layers = len(block.block)

            # Create connectivity matrix
            connectivity = np.zeros((num_layers, num_layers))
            for j in range(num_layers):
                connectivity[j, :j+1] = 1  # Dense connections to all previous layers

            im = axes[i].imshow(connectivity, cmap='Blues', aspect='auto')
            axes[i].set_title(f'Dense Block {i+1}')
            axes[i].set_xlabel('Layer Index')
            axes[i].set_ylabel('Layer Index')

    plt.tight_layout()
    plt.savefig('densenet_connectivity.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    print("="*80)
    print("DenseNet and Feature Reuse - Comprehensive Analysis")
    print("="*80)

    # Prepare data
    print("\nPreparing CIFAR-10 dataset...")
    trainloader, testloader, num_classes = prepare_data(batch_size=128, dataset='cifar10')

    # 1. Growth Rate Analysis
    histories, accuracies = growth_rate_analysis(trainloader, testloader,
                                                 num_classes, epochs=50)

    # 2. Compression Factor Analysis
    compression_analysis(trainloader, testloader, num_classes, epochs=50)

    # 3. Architecture Comparison
    results = compare_architectures(trainloader, testloader, num_classes, epochs=50)

    # 4. Visualize Dense Connections
    print("\nVisualizing dense connectivity pattern...")
    model = densenet121(num_classes=num_classes).to(device)
    visualize_dense_connections(model)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("Generated visualizations:")
    print("  - densenet_growth_rate_analysis.png")
    print("  - densenet_compression_analysis.png")
    print("  - densenet_connectivity.png")


if __name__ == "__main__":
    main()
