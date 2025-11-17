"""
Learning Rate Schedules - Deep Learning Solution

Comprehensive implementation demonstrating advanced techniques.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomModel(nn.Module):
    """Custom neural network model"""
    def __init__(self, input_size=32*32*3, hidden_sizes=[512, 256, 128], num_classes=10):
        super().__init__()
        layers = []
        in_size = input_size
        
        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_size = h_size
        
        layers.append(nn.Linear(in_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


class CNNModel(nn.Module):
    """Convolutional neural network"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


def prepare_data(batch_size=128, augment=True):
    """Prepare CIFAR-10 dataset"""
    
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([
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
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    
    return trainloader, testloader


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
    """Train model with comprehensive tracking"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'learning_rates': []
    }
    
    best_acc = 0
    best_epoch = 0
    
    print(f"\nTraining {name}...")
    print("="*80)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # Validation
        test_loss, test_acc, _, _ = evaluate(model, testloader, criterion, device)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(current_lr)
        
        # Track best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
        
        epoch_time = time.time() - start_time
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
    
    print("="*80)
    print(f"Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print("="*80)
    
    return history, best_acc


def plot_training_history(histories, names, save_path='training_curves.png'):
    """Plot training curves for multiple experiments"""
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")


def run_experiments(epochs=50):
    """Run comprehensive experiments"""
    print("="*80)
    print(f"{title} - Comprehensive Experiments")
    print("="*80)
    
    # Prepare data
    trainloader, testloader = prepare_data(batch_size=128, augment=True)
    
    # Experiment configurations
    experiments = [
        (CNNModel(num_classes=10), "CNN Baseline"),
        (CustomModel(num_classes=10), "Custom Model"),
    ]
    
    histories = []
    names = []
    results = []
    
    for model, name in experiments:
        model = model.to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{name}: {num_params:,} parameters")
        
        # Train model
        history, best_acc = train_model(model, trainloader, testloader,
                                       epochs=epochs, lr=0.1, name=name)
        
        histories.append(history)
        names.append(name)
        results.append((name, num_params, best_acc))
    
    # Plot comparisons
    plot_training_history(histories, names)
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Parameters':<20} {'Best Accuracy':<15}")
    print("-"*80)
    for name, params, acc in results:
        print(f"{name:<30} {params:>15,} {acc:>12.2f}%")
    print("="*80)
    
    return histories, results




class AdvancedCNN(nn.Module):
    """Advanced CNN with modern techniques"""
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        
        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Fourth block
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AttentionModule(nn.Module):
    """Attention mechanism for CNNs"""
    def __init__(self, in_channels):
        super().__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        
        return x


def calculate_metrics(y_true, y_pred, num_classes=10):
    """Calculate comprehensive metrics"""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, classification_report)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                average='weighted',
                                                                zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'confusion_matrix': cm
    }
    
    return metrics


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_predictions(model, testloader, device):
    """Analyze model predictions in detail"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Find misclassified samples
    misclassified_idx = np.where(all_preds != all_targets)[0]
    print(f"\nMisclassified samples: {len(misclassified_idx)} / {len(all_targets)}")
    
    return metrics, all_probs


def hyperparameter_search(trainloader, testloader, epochs=30):
    """Search for optimal hyperparameters"""
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH")
    print("="*80)
    
    # Learning rate search
    learning_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0.3, 0.5]
    
    best_config = None
    best_acc = 0
    results = []
    
    for lr in learning_rates:
        for dropout in dropout_rates:
            print(f"\nTesting LR={lr}, Dropout={dropout}")
            
            model = AdvancedCNN(num_classes=10, dropout_rate=dropout).to(device)
            history, acc = train_model(model, trainloader, testloader,
                                      epochs=epochs, lr=lr,
                                      name=f'LR={lr}, Dropout={dropout}')
            
            results.append((lr, dropout, acc))
            
            if acc > best_acc:
                best_acc = acc
                best_config = (lr, dropout)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    print(f"{'Learning Rate':<20} {'Dropout':<15} {'Accuracy':<15}")
    print("-"*80)
    for lr, dropout, acc in results:
        marker = " <-- BEST" if (lr, dropout) == best_config else ""
        print(f"{lr:<20} {dropout:<15} {acc:>10.2f}%{marker}")
    print("="*80)
    
    return best_config, results


def cross_validation_analysis(batch_size=128, epochs=30, k_folds=5):
    """Perform k-fold cross-validation"""
    print("\n" + "="*80)
    print(f"{k_folds}-FOLD CROSS-VALIDATION")
    print("="*80)
    
    # Load full dataset
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    
    fold_size = len(full_dataset) // k_folds
    fold_accuracies = []
    
    for fold in range(k_folds):
        print(f"\nFold {fold + 1}/{k_folds}")
        print("-"*80)
        
        # Create train/val split
        val_start = fold * fold_size
        val_end = val_start + fold_size
        
        train_indices = list(range(0, val_start)) + list(range(val_end, len(full_dataset)))
        val_indices = list(range(val_start, val_end))
        
        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)
        
        trainloader = DataLoader(train_subset, batch_size=batch_size,
                                shuffle=True, num_workers=2)
        valloader = DataLoader(val_subset, batch_size=batch_size,
                              shuffle=False, num_workers=2)
        
        # Train model
        model = CNNModel(num_classes=10).to(device)
        history, acc = train_model(model, trainloader, valloader,
                                  epochs=epochs, lr=0.1, name=f'Fold {fold+1}')
        
        fold_accuracies.append(acc)
    
    # Summary
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.2f}%")
    print("-"*80)
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("="*80)
    
    return fold_accuracies


def ensemble_predictions(models, testloader, device):
    """Make ensemble predictions from multiple models"""
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTION")
    print("="*80)
    
    all_outputs = []
    
    for i, model in enumerate(models):
        model.eval()
        outputs_list = []
        
        with torch.no_grad():
            for inputs, _ in testloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                outputs_list.append(probs)
        
        all_outputs.append(torch.cat(outputs_list, dim=0))
    
    # Average predictions
    ensemble_probs = torch.stack(all_outputs).mean(dim=0)
    ensemble_preds = ensemble_probs.argmax(dim=1)
    
    # Calculate accuracy
    all_targets = []
    for _, targets in testloader:
        all_targets.extend(targets.numpy())
    all_targets = np.array(all_targets)
    
    ensemble_acc = 100. * (ensemble_preds.cpu().numpy() == all_targets).sum() / len(all_targets)
    
    print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")
    print("="*80)
    
    return ensemble_acc, ensemble_preds




def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    
    # Run experiments
    histories, results = run_experiments(epochs=50)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("Generated files:")
    print("  - training_curves.png")
    print("="*80)


if __name__ == "__main__":
    main()
