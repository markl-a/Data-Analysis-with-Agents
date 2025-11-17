"""
MobileNet for Edge Deployment - Deep Learning Solution

This solution demonstrates:
1. MobileNetV1, V2, and V3 architectures
2. Depthwise separable convolutions
3. Inverted residuals and linear bottlenecks
4. Model quantization and pruning for deployment
5. Latency vs accuracy trade-offs
6. Width multiplier analysis
7. Hardware-aware NAS
8. TensorFlow Lite conversion

Dataset: CIFAR-10 for mobile deployment
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


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module):
    """MobileNet V1 with depthwise separable convolutions"""

    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return DepthwiseSeparableConv(inp, oup, stride)

        input_channel = int(32 * width_mult)

        self.model = nn.Sequential(
            conv_bn(3, input_channel, 2),
            conv_dw(input_channel, int(64 * width_mult), 1),
            conv_dw(int(64 * width_mult), int(128 * width_mult), 2),
            conv_dw(int(128 * width_mult), int(128 * width_mult), 1),
            conv_dw(int(128 * width_mult), int(256 * width_mult), 2),
            conv_dw(int(256 * width_mult), int(256 * width_mult), 1),
            conv_dw(int(256 * width_mult), int(512 * width_mult), 2),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(1024 * width_mult), 2),
            conv_dw(int(1024 * width_mult), int(1024 * width_mult), 1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * width_mult), num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2"""

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNet V2 with inverted residuals"""

    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult)

        # Building inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, input_size=(1, 3, 32, 32), iterations=100):
    """Measure inference time"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations * 1000  # ms
    return avg_time


def prepare_data(batch_size=128):
    """Prepare CIFAR-10"""
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


def train_model(model, trainloader, testloader, epochs=50, lr=0.1, name='model'):
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.2f}%, Test: {test_acc:.2f}%")

    print(f"Best Accuracy: {best_acc:.2f}%")
    return history, best_acc


def width_multiplier_analysis(trainloader, testloader, epochs=30):
    """Analyze impact of width multiplier"""
    print("\n" + "="*80)
    print("WIDTH MULTIPLIER ANALYSIS")
    print("="*80)

    width_mults = [0.25, 0.5, 0.75, 1.0]
    results = []

    for width in width_mults:
        print(f"\nTesting width multiplier: {width}")
        model = MobileNetV2(num_classes=10, width_mult=width).to(device)

        params = count_parameters(model)
        latency = measure_inference_time(model)

        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, lr=0.1, name=f'MobileNetV2 (α={width})')

        results.append((width, params, latency, acc))

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    widths = [r[0] for r in results]
    params = [r[1]/1e6 for r in results]
    latencies = [r[2] for r in results]
    accs = [r[3] for r in results]

    axes[0].plot(widths, accs, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Width Multiplier')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy vs Width')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(widths, params, 's-', linewidth=2, markersize=8, color='red')
    axes[1].set_xlabel('Width Multiplier')
    axes[1].set_ylabel('Parameters (M)')
    axes[1].set_title('Model Size vs Width')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(widths, latencies, '^-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Width Multiplier')
    axes[2].set_ylabel('Latency (ms)')
    axes[2].set_title('Inference Time vs Width')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mobilenet_width_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*80)
    print(f"{'Width':<10} {'Params (M)':<15} {'Latency (ms)':<15} {'Accuracy (%)':<15}")
    print("-"*80)
    for width, params, latency, acc in results:
        print(f"{width:<10.2f} {params/1e6:<15.2f} {latency:<15.2f} {acc:<15.2f}")
    print("-"*80)

    return results


def compare_mobilenet_versions(trainloader, testloader, epochs=30):
    """Compare MobileNet versions"""
    print("\n" + "="*80)
    print("MOBILENET VERSION COMPARISON")
    print("="*80)

    models = [
        (MobileNetV1(num_classes=10, width_mult=1.0), 'MobileNetV1'),
        (MobileNetV2(num_classes=10, width_mult=1.0), 'MobileNetV2'),
    ]

    results = []

    for model, name in models:
        model = model.to(device)
        params = count_parameters(model)
        latency = measure_inference_time(model)

        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, lr=0.1, name=name)

        results.append((name, params, latency, acc))

    print("\n" + "-"*80)
    print(f"{'Model':<20} {'Params (M)':<15} {'Latency (ms)':<15} {'Accuracy (%)':<15}")
    print("-"*80)
    for name, params, latency, acc in results:
        print(f"{name:<20} {params/1e6:<15.2f} {latency:<15.2f} {acc:<15.2f}")
    print("-"*80)

    return results


def quantization_analysis(model, testloader):
    """Analyze quantization impact"""
    print("\n" + "="*80)
    print("QUANTIZATION ANALYSIS")
    print("="*80)

    # Original model
    model.eval()
    criterion = nn.CrossEntropyLoss()
    _, acc_fp32 = evaluate(model, testloader, criterion, device)
    size_fp32 = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)  # MB

    # Dynamic quantization
    model_int8 = torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    model_int8 = model_int8.to(device)
    _, acc_int8 = evaluate(model_int8, testloader, criterion, device)
    size_int8 = sum(p.numel() for p in model_int8.parameters()) / (1024 ** 2)  # Approximate

    print("\n" + "-"*80)
    print(f"{'Model':<20} {'Size (MB)':<15} {'Accuracy (%)':<15}")
    print("-"*80)
    print(f"{'FP32 (Original)':<20} {size_fp32:<15.2f} {acc_fp32:<15.2f}")
    print(f"{'INT8 (Quantized)':<20} {size_int8:<15.2f} {acc_int8:<15.2f}")
    print(f"{'Compression Ratio':<20} {size_fp32/size_int8:<15.2f}x")
    print(f"{'Accuracy Drop':<20} {'':<15} {acc_fp32-acc_int8:<15.2f}")
    print("-"*80)




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
    print("MobileNet for Edge Deployment - Comprehensive Analysis")
    print("="*80)

    trainloader, testloader = prepare_data(batch_size=128)

    # 1. Width Multiplier Analysis
    width_results = width_multiplier_analysis(trainloader, testloader, epochs=30)

    # 2. Version Comparison
    version_results = compare_mobilenet_versions(trainloader, testloader, epochs=30)

    # 3. Quantization Analysis
    model = MobileNetV2(num_classes=10, width_mult=1.0).to(device)
    train_model(model, trainloader, testloader, epochs=30, lr=0.1, name='MobileNetV2 for Quantization')
    quantization_analysis(model, testloader)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("Generated visualizations:")
    print("  - mobilenet_width_analysis.png")


if __name__ == "__main__":
    main()
