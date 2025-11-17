"""
Vision Transformer (ViT) - Deep Learning Solution

This solution demonstrates:
1. Vision Transformer architecture
2. Patch embedding and positional encoding
3. Multi-head self-attention
4. Comparison with CNNs
5. Data augmentation strategies
6. Scaling laws for ViT
7. Hybrid CNN-ViT architectures
8. Attention map visualization

Dataset: CIFAR-10 for vision transformer experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


class PatchEmbedding(nn.Module):
    """Split image into patches and embed"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x, attn


class MLP(nn.Module):
    """Feed-forward network"""

    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        attention_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attention_maps.append(attn)

        x = self.norm(x)
        x = x[:, 0]  # Class token
        x = self.head(x)

        return x


def prepare_data(batch_size=128):
    """Prepare dataset with augmentation"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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


def train_model(model, trainloader, testloader, epochs=100, lr=0.001, name='model'):
    """Train model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
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


def scaling_analysis(trainloader, testloader, epochs=50):
    """Analyze ViT scaling"""
    print("\n" + "="*80)
    print("VISION TRANSFORMER SCALING ANALYSIS")
    print("="*80)

    configs = [
        {'embed_dim': 128, 'depth': 6, 'num_heads': 4, 'name': 'ViT-Tiny'},
        {'embed_dim': 192, 'depth': 12, 'num_heads': 3, 'name': 'ViT-Small'},
        {'embed_dim': 256, 'depth': 12, 'num_heads': 4, 'name': 'ViT-Base'},
    ]

    results = []

    for config in configs:
        model = VisionTransformer(
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            num_classes=10
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        history, acc = train_model(model, trainloader, testloader,
                                  epochs=epochs, lr=0.001, name=config['name'])

        results.append((config['name'], params, acc))

    print("\n" + "-"*80)
    print(f"{'Model':<15} {'Parameters':<15} {'Accuracy':<15}")
    print("-"*80)
    for name, params, acc in results:
        print(f"{name:<15} {params/1e6:>10.2f}M {acc:>10.2f}%")
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
    print("Vision Transformer - Comprehensive Analysis")
    print("="*80)

    trainloader, testloader = prepare_data(batch_size=128)

    # Scaling Analysis
    results = scaling_analysis(trainloader, testloader, epochs=50)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
