"""
Dropout Variants (Spatial, DropConnect, Cutout) - Deep Learning Solution

This solution demonstrates:
1. Standard dropout
2. Spatial dropout for CNNs
3. DropConnect
4. Cutout and random erasing
5. DropBlock
6. Stochastic depth
7. Dropout scheduling
8. Ablation studies on different dropout methods

Dataset: CIFAR-10/CIFAR-100 for image classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpatialDropout(nn.Module):
    """Spatial Dropout - drops entire feature maps"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # x shape: (batch, channels, height, width)
        batch_size, channels, height, width = x.size()

        # Create dropout mask for entire channels
        mask = torch.bernoulli(torch.full((batch_size, channels, 1, 1), 1 - self.p, device=x.device))
        mask = mask.expand_as(x)

        return x * mask / (1 - self.p)


class DropConnect(nn.Module):
    """DropConnect - drops random weights"""

    def __init__(self, in_features, out_features, p=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.bernoulli(torch.full_like(self.weight, 1 - self.p))
            weight = self.weight * mask / (1 - self.p)
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)


class DropBlock(nn.Module):
    """DropBlock - drops contiguous regions"""

    def __init__(self, block_size=7, p=0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        gamma = self._compute_gamma(x)
        mask = torch.bernoulli(torch.full_like(x, gamma))

        block_mask = F.max_pool2d(mask, kernel_size=self.block_size,
                                   stride=1, padding=self.block_size // 2)
        block_mask = 1 - block_mask

        normalize_factor = block_mask.numel() / block_mask.sum()
        return x * block_mask * normalize_factor

    def _compute_gamma(self, x):
        return self.p / (self.block_size ** 2)


class Cutout(object):
    """Cutout data augmentation"""

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class StochasticDepth(nn.Module):
    """Stochastic Depth - randomly drops layers"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x, residual):
        if not self.training:
            return x + residual

        if torch.rand(1).item() < self.p:
            return x  # Skip residual
        return x + residual


class CNNWithStandardDropout(nn.Module):
    """CNN with standard dropout"""

    def __init__(self, num_classes=10, dropout_p=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNNWithSpatialDropout(nn.Module):
    """CNN with spatial dropout"""

    def __init__(self, num_classes=10, dropout_p=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SpatialDropout(dropout_p * 0.5),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SpatialDropout(dropout_p * 0.5),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SpatialDropout(dropout_p * 0.5),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNNWithDropBlock(nn.Module):
    """CNN with DropBlock"""

    def __init__(self, num_classes=10, dropout_p=0.1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DropBlock(block_size=5, p=dropout_p),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DropBlock(block_size=5, p=dropout_p),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DropBlock(block_size=5, p=dropout_p),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def prepare_data(batch_size=128, use_cutout=False):
    """Prepare CIFAR-10 dataset"""

    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    if use_cutout:
        transform_list.append(Cutout(n_holes=1, length=16))

    transform_train = transforms.Compose(transform_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
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


def train_model(model, trainloader, testloader, epochs=100, lr=0.1, name='model'):
    """Train model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0

    print(f"\nTraining {name}...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.2f}%, Test: {test_acc:.2f}%")

    print(f"Best Accuracy: {best_acc:.2f}%")
    return history, best_acc


def compare_dropout_methods(trainloader_no_cutout, trainloader_cutout, testloader, epochs=50):
    """Compare different dropout methods"""
    print("\n" + "="*80)
    print("DROPOUT METHODS COMPARISON")
    print("="*80)

    models = [
        (CNNWithStandardDropout(num_classes=10, dropout_p=0.5), trainloader_no_cutout, 'Standard Dropout'),
        (CNNWithSpatialDropout(num_classes=10, dropout_p=0.5), trainloader_no_cutout, 'Spatial Dropout'),
        (CNNWithDropBlock(num_classes=10, dropout_p=0.1), trainloader_no_cutout, 'DropBlock'),
        (CNNWithStandardDropout(num_classes=10, dropout_p=0.5), trainloader_cutout, 'Standard + Cutout'),
    ]

    results = []

    for model, trainloader, name in models:
        model = model.to(device)
        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, lr=0.1, name=name)
        results.append((name, acc, history))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for name, acc, history in results:
        axes[0].plot(history['train_acc'], label=name)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Accuracy (%)')
    axes[0].set_title('Training Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, acc, history in results:
        axes[1].plot(history['test_acc'], label=name)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dropout_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*80)
    print(f"{'Method':<25} {'Test Accuracy':<15}")
    print("-"*80)
    for name, acc, _ in results:
        print(f"{name:<25} {acc:>10.2f}%")
    print("-"*80)

    return results


def dropout_rate_analysis(testloader, epochs=50):
    """Analyze impact of dropout rate"""
    print("\n" + "="*80)
    print("DROPOUT RATE ANALYSIS")
    print("="*80)

    dropout_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
    results = []

    for p in dropout_rates:
        trainloader, _ = prepare_data(batch_size=128, use_cutout=False)
        model = CNNWithStandardDropout(num_classes=10, dropout_p=p).to(device)

        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, lr=0.1, name=f'Dropout p={p}')

        results.append((p, acc))

    # Visualization
    plt.figure(figsize=(10, 6))
    ps = [r[0] for r in results]
    accs = [r[1] for r in results]

    plt.plot(ps, accs, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Dropout Rate')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Impact of Dropout Rate on Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dropout_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*80)
    print(f"{'Dropout Rate':<20} {'Test Accuracy':<15}")
    print("-"*80)
    for p, acc in results:
        print(f"{p:<20.1f} {acc:>10.2f}%")
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
    print("Dropout Variants - Comprehensive Analysis")
    print("="*80)

    trainloader_no_cutout, testloader = prepare_data(batch_size=128, use_cutout=False)
    trainloader_cutout, _ = prepare_data(batch_size=128, use_cutout=True)

    # 1. Compare Dropout Methods
    method_results = compare_dropout_methods(trainloader_no_cutout, trainloader_cutout,
                                            testloader, epochs=50)

    # 2. Dropout Rate Analysis
    rate_results = dropout_rate_analysis(testloader, epochs=50)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("Generated visualizations:")
    print("  - dropout_comparison.png")
    print("  - dropout_rate_analysis.png")


if __name__ == "__main__":
    main()
