"""
EfficientNet and Compound Scaling - Deep Learning Solution

This solution demonstrates:
1. EfficientNet architecture (B0-B7)
2. Compound scaling (depth, width, resolution)
3. Mobile Inverted Bottleneck Convolution (MBConv)
4. Squeeze-and-Excitation blocks
5. Scaling coefficient analysis
6. Resource-constrained optimization
7. AutoML-derived architectures
8. Ablation studies on scaling dimensions

Dataset: CIFAR-10/ImageNet for image classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Dict
import time
import math
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SwishActivation(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            SwishActivation(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate

        hidden_dim = in_channels * expand_ratio
        reduced_dim = max(1, int(in_channels * se_ratio))

        layers = []

        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SwishActivation()
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SwishActivation()
        ])

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(hidden_dim, reduced_dim)

        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)

    def _drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep_prob = 1 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand(
            [x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask


class EfficientNet(nn.Module):
    """EfficientNet architecture with compound scaling"""

    def __init__(self, width_mult=1.0, depth_mult=1.0, resolution=224,
                 num_classes=1000, dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        # Base configuration for EfficientNet-B0
        # [expand_ratio, channels, num_blocks, stride, kernel_size]
        base_config = [
            [1, 16, 1, 1, 3],   # Stage 1
            [6, 24, 2, 2, 3],   # Stage 2
            [6, 40, 2, 2, 5],   # Stage 3
            [6, 80, 3, 2, 3],   # Stage 4
            [6, 112, 3, 1, 5],  # Stage 5
            [6, 192, 4, 2, 5],  # Stage 6
            [6, 320, 1, 1, 3],  # Stage 7
        ]

        # Stem
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SwishActivation()
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        in_channels = out_channels

        for expand_ratio, channels, num_blocks, stride, kernel_size in base_config:
            out_channels = self._round_filters(channels, width_mult)
            num_blocks = self._round_repeats(num_blocks, depth_mult)

            for i in range(num_blocks):
                self.blocks.append(
                    MBConvBlock(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size,
                        stride if i == 0 else 1,
                        expand_ratio
                    )
                )

        # Head
        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            SwishActivation()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(final_channels, num_classes)

        self._initialize_weights()

    def _round_filters(self, filters, width_mult):
        """Round number of filters based on width multiplier"""
        filters = int(filters * width_mult)
        return max(8, filters)

    def _round_repeats(self, repeats, depth_mult):
        """Round number of repeats based on depth multiplier"""
        return int(math.ceil(depth_mult * repeats))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_efficientnet_params(version='b0'):
    """Get EfficientNet parameters for different versions"""
    params = {
        'b0': (1.0, 1.0, 224, 0.2),
        'b1': (1.0, 1.1, 240, 0.2),
        'b2': (1.1, 1.2, 260, 0.3),
        'b3': (1.2, 1.4, 300, 0.3),
        'b4': (1.4, 1.8, 380, 0.4),
        'b5': (1.6, 2.2, 456, 0.4),
        'b6': (1.8, 2.6, 528, 0.5),
        'b7': (2.0, 3.1, 600, 0.5),
    }
    width_mult, depth_mult, resolution, dropout = params[version]
    return width_mult, depth_mult, resolution, dropout


def efficientnet_b0(num_classes=10):
    """EfficientNet-B0"""
    width, depth, res, dropout = get_efficientnet_params('b0')
    return EfficientNet(width, depth, res, num_classes, dropout)


def efficientnet_b1(num_classes=10):
    """EfficientNet-B1"""
    width, depth, res, dropout = get_efficientnet_params('b1')
    return EfficientNet(width, depth, res, num_classes, dropout)


def efficientnet_b2(num_classes=10):
    """EfficientNet-B2"""
    width, depth, res, dropout = get_efficientnet_params('b2')
    return EfficientNet(width, depth, res, num_classes, dropout)


def prepare_data(batch_size=128, resolution=224):
    """Prepare CIFAR-10 dataset with specified resolution"""

    transform_train = transforms.Compose([
        transforms.Resize(resolution),
        transforms.RandomCrop(resolution, padding=resolution//8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    return trainloader, testloader, 10


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

    return running_loss / len(dataloader), 100. * correct / total


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

    return running_loss / len(dataloader), 100. * correct / total


def train_model(model, trainloader, testloader, epochs=50, lr=0.1, name='model'):
    """Train model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.2f}%, "
                  f"Test: {test_acc:.2f}%, Time: {time.time()-start_time:.2f}s")

    print(f"Best Accuracy: {best_acc:.2f}%")
    return history, best_acc


def scaling_dimension_analysis(num_classes=10, epochs=30):
    """Analyze impact of each scaling dimension"""
    print("\n" + "="*80)
    print("SCALING DIMENSION ANALYSIS")
    print("="*80)

    # Baseline (B0)
    baseline_trainloader, baseline_testloader, _ = prepare_data(batch_size=64, resolution=224)

    results = []

    # 1. Baseline (no scaling)
    model = EfficientNet(1.0, 1.0, 224, num_classes, 0.2).to(device)
    history, acc = train_model(model, baseline_trainloader, baseline_testloader,
                              epochs=epochs, lr=0.01, name='Baseline (w=1.0, d=1.0, r=224)')
    results.append(('Baseline', 1.0, 1.0, 224, acc))

    # 2. Width scaling only
    model = EfficientNet(1.2, 1.0, 224, num_classes, 0.2).to(device)
    history, acc = train_model(model, baseline_trainloader, baseline_testloader,
                              epochs=epochs, lr=0.01, name='Width Scaled (w=1.2, d=1.0, r=224)')
    results.append(('Width Only', 1.2, 1.0, 224, acc))

    # 3. Depth scaling only
    model = EfficientNet(1.0, 1.2, 224, num_classes, 0.2).to(device)
    history, acc = train_model(model, baseline_trainloader, baseline_testloader,
                              epochs=epochs, lr=0.01, name='Depth Scaled (w=1.0, d=1.2, r=224)')
    results.append(('Depth Only', 1.0, 1.2, 224, acc))

    # 4. Resolution scaling only
    res_trainloader, res_testloader, _ = prepare_data(batch_size=64, resolution=260)
    model = EfficientNet(1.0, 1.0, 260, num_classes, 0.2).to(device)
    history, acc = train_model(model, res_trainloader, res_testloader,
                              epochs=epochs, lr=0.01, name='Resolution Scaled (w=1.0, d=1.0, r=260)')
    results.append(('Resolution Only', 1.0, 1.0, 260, acc))

    # 5. Compound scaling
    model = EfficientNet(1.2, 1.2, 260, num_classes, 0.2).to(device)
    history, acc = train_model(model, res_trainloader, res_testloader,
                              epochs=epochs, lr=0.01, name='Compound Scaled (w=1.2, d=1.2, r=260)')
    results.append(('Compound', 1.2, 1.2, 260, acc))

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r[0] for r in results]
    accs = [r[4] for r in results]

    bars = ax.bar(names, accs, color=['gray', 'blue', 'green', 'red', 'purple'])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Impact of Different Scaling Dimensions')
    ax.set_ylim([min(accs) - 5, max(accs) + 5])

    for i, (name, acc) in enumerate(zip(names, accs)):
        ax.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('efficientnet_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*80)
    print("SCALING RESULTS:")
    for name, width, depth, res, acc in results:
        print(f"{name:20s} w={width:.1f}, d={depth:.1f}, r={res:3d} -> {acc:.2f}%")
    print("-"*80)

    return results


def compare_efficientnet_versions(num_classes=10, epochs=30):
    """Compare different EfficientNet versions"""
    print("\n" + "="*80)
    print("EFFICIENTNET VERSION COMPARISON")
    print("="*80)

    versions = ['b0', 'b1', 'b2']
    results = []

    for version in versions:
        width, depth, res, dropout = get_efficientnet_params(version)
        trainloader, testloader, _ = prepare_data(batch_size=64, resolution=res)

        model = EfficientNet(width, depth, res, num_classes, dropout).to(device)
        params = sum(p.numel() for p in model.parameters())

        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, lr=0.01, name=f'EfficientNet-{version.upper()}')

        results.append((version.upper(), params, acc))

    print("\n" + "-"*80)
    print(f"{'Version':<15} {'Parameters':<15} {'Accuracy':<15}")
    print("-"*80)
    for version, params, acc in results:
        print(f"{version:<15} {params/1e6:>10.2f}M {acc:>10.2f}%")
    print("-"*80)

    return results




def advanced_data_augmentation():
    """Advanced data augmentation strategies"""
    from torchvision.transforms import autoaugment
    
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.CIFAR10),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


class MixupDataset:
    """Mixup data augmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def mixup_data(self, x, y):
        """Apply mixup to batch"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup loss"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_score = None
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints"""
    def __init__(self, filepath, monitor='val_loss', mode='min', verbose=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = None
    
    def __call__(self, model, metric):
        if self.best is None:
            self.best = metric
            self.save_model(model)
        else:
            if self.mode == 'min':
                if metric < self.best:
                    self.best = metric
                    self.save_model(model)
            else:
                if metric > self.best:
                    self.best = metric
                    self.save_model(model)
    
    def save_model(self, model):
        torch.save(model.state_dict(), self.filepath)
        if self.verbose:
            print(f'Model checkpoint saved to {self.filepath}')


def learning_rate_finder(model, trainloader, device, start_lr=1e-7, end_lr=10, num_iter=100):
    """Find optimal learning rate"""
    print("\n" + "="*80)
    print("LEARNING RATE FINDER")
    print("="*80)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
    
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    lrs = []
    losses = []
    best_loss = 1e9
    
    iterator = iter(trainloader)
    for iteration in range(num_iter):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(trainloader)
            inputs, targets = next(iterator)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Track
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # Stop if loss explodes
        if loss.item() > 4 * best_loss:
            break
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_finder.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Suggest LR
    min_loss_idx = np.argmin(losses[:len(losses)//2])  # Look at first half
    suggested_lr = lrs[min_loss_idx] / 10  # Use 1/10th of min loss LR
    
    print(f"Suggested learning rate: {suggested_lr:.6f}")
    print("="*80)
    
    return suggested_lr, lrs, losses


def gradcam_visualization(model, input_tensor, target_layer, class_idx=None):
    """Generate Grad-CAM visualization"""
    model.eval()
    
    # Forward pass
    output = model(input_tensor)
    
    if class_idx is None:
        class_idx = output.argmax(dim=1)
    
    # Backward pass
    model.zero_grad()
    class_loss = output[0, class_idx]
    class_loss.backward()
    
    # Get gradients and activations
    gradients = target_layer.weight.grad
    activations = target_layer.weight.data
    
    # Weight the channels by gradient
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    
    # ReLU and normalize
    cam = F.relu(cam)
    cam = cam / torch.max(cam)
    
    return cam


def calculate_flops(model, input_size=(1, 3, 32, 32)):
    """Calculate FLOPs for model"""
    def count_conv2d(layer, input_shape):
        """Count FLOPs for Conv2d layer"""
        batch_size, in_c, in_h, in_w = input_shape
        out_c = layer.out_channels
        k_h, k_w = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
        
        out_h = (in_h + 2 * layer.padding[0] - k_h) // layer.stride[0] + 1
        out_w = (in_w + 2 * layer.padding[1] - k_w) // layer.stride[1] + 1
        
        flops = out_h * out_w * out_c * (in_c * k_h * k_w)
        return flops, (batch_size, out_c, out_h, out_w)
    
    def count_linear(layer, input_shape):
        """Count FLOPs for Linear layer"""
        return layer.in_features * layer.out_features, input_shape
    
    total_flops = 0
    x = torch.randn(input_size)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            flops, x_shape = count_conv2d(module, x.shape)
            total_flops += flops
        elif isinstance(module, nn.Linear):
            flops, _ = count_linear(module, x.shape)
            total_flops += flops
    
    return total_flops


def train_with_mixed_precision(model, trainloader, testloader, epochs=50, lr=0.1):
    """Train with mixed precision for faster training"""
    print("\n" + "="*80)
    print("TRAINING WITH MIXED PRECISION")
    print("="*80)
    
    from torch.cuda.amp import autocast, GradScaler
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Validation
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.2f}%, Test: {test_acc:.2f}%")
    
    print("="*80)
    return history



def main():
    """Main execution"""
    print("="*80)
    print("EfficientNet and Compound Scaling - Comprehensive Analysis")
    print("="*80)

    # 1. Scaling Dimension Analysis
    scaling_results = scaling_dimension_analysis(num_classes=10, epochs=30)

    # 2. EfficientNet Version Comparison
    version_results = compare_efficientnet_versions(num_classes=10, epochs=30)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("Generated visualizations:")
    print("  - efficientnet_scaling_analysis.png")


if __name__ == "__main__":
    main()
