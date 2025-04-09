import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import os
import gdown

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Constants
NUM_CLASSES = 100
BATCH_SIZE = 128
LATENT_DIM = 100
IMAGE_SIZE = 32
CHANNELS = 3
TEACHER_WEIGHTS_URL = "https://drive.google.com/uc?id=1iKJjnxJn5Qp1SQ_hD-J9hSqsV1JvIwE3"
TEACHER_WEIGHTS_PATH = 'weights/best_resnet34_cifar100.pth'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download teacher model weights if not available
def download_teacher_weights():
    if not os.path.exists(TEACHER_WEIGHTS_PATH):
        print("Downloading teacher model weights...")
        os.makedirs('weights', exist_ok=True)
        gdown.download(TEACHER_WEIGHTS_URL, TEACHER_WEIGHTS_PATH, quiet=False)
        print(f"Teacher model weights downloaded to {TEACHER_WEIGHTS_PATH}")
    else:
        print(f"Teacher model weights already exist at {TEACHER_WEIGHTS_PATH}")

class SmallStudent(nn.Module):
    """Student model with ~10% of teacher parameters (~2.1M params)"""
    def __init__(self, num_classes=NUM_CLASSES):
        super(SmallStudent, self).__init__()
        
        self.features = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MediumStudent(nn.Module):
    """Student model with ~20% of teacher parameters (~4.2M params)"""
    def __init__(self, num_classes=NUM_CLASSES):
        super(MediumStudent, self).__init__()
        
        self.features = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Generator(nn.Module):
    """Generator network for creating synthetic images"""
    def __init__(self, latent_dim=LATENT_DIM):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Initial dense layer
            nn.Linear(latent_dim, 4*4*512),
            nn.BatchNorm1d(4*4*512),
            nn.ReLU(True),
            
            # Reshape
            nn.Unflatten(1, (512, 4, 4)),
            
            # Transposed convolutions
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, CHANNELS, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

def load_teacher_model():
    """Load the pre-trained ResNet-34 teacher model"""
    
    model = torchvision.models.resnet34(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(TEACHER_WEIGHTS_PATH))
    model = model.to(device)
    model.eval()
    return model

def get_dataloaders(split_ratio):
    """Create test dataloaders with given split ratio"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Calculate split sizes
    test_size = len(test_dataset)
    split_size = int(test_size * split_ratio)
    
    # Create splits
    indices = list(range(test_size))
    np.random.shuffle(indices)
    split_indices = indices[:split_size]
    
    # Create subset
    split_dataset = Subset(test_dataset, split_indices)
    
    # Create dataloader
    split_loader = DataLoader(
        split_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    return split_loader

def train_step(teacher, student, generator, batch_size, optimizers):
    """Single training step for AKD"""
    student.train()
    generator.train()
    
    # Optimizers
    opt_student, opt_generator = optimizers
    
    # Generate synthetic images
    z = torch.randn(batch_size, LATENT_DIM).to(device)
    synthetic_images = generator(z)
    
    # Get predictions
    with torch.no_grad():
        teacher_logits = teacher(synthetic_images)
    student_logits = student(synthetic_images)
    
    # Calculate losses
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / 2.0, dim=1),
        nn.functional.softmax(teacher_logits / 2.0, dim=1)
    )
    
    # Generator loss (encourage diverse and realistic images)
    generator_loss = -distillation_loss
    
    # Update student
    opt_student.zero_grad()
    distillation_loss.backward(retain_graph=True)
    opt_student.step()
    
    # Update generator
    opt_generator.zero_grad()
    generator_loss.backward()
    opt_generator.step()
    
    return distillation_loss.item(), generator_loss.item()

def evaluate(model, dataloader):
    """Evaluate model on given dataloader"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, conf_matrix

def plot_results(student_name, accuracy, conf_matrix, generated_images):
    """Plot and save evaluation results"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=False)
    plt.title(f'{student_name} Confusion Matrix\nAccuracy: {accuracy:.2f}%')
    plt.savefig(f'results/{student_name}_confusion_matrix.png')
    plt.close()
    
    # Plot generated images
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(generated_images))):
        plt.subplot(5, 5, i+1)
        plt.imshow(generated_images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(f'results/{student_name}_generated_images.png')
    plt.close()

def main():
    # Load teacher model
    teacher = load_teacher_model()
    print("Teacher model loaded")
    
    # Initialize models
    small_student = SmallStudent().to(device)
    medium_student = MediumStudent().to(device)
    generator = Generator().to(device)
    
    # Print model parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Teacher parameters: {count_parameters(teacher):,}")
    print(f"Small Student parameters: {count_parameters(small_student):,}")
    print(f"Medium Student parameters: {count_parameters(medium_student):,}")
    
    # Training configuration
    num_epochs = 100
    student_lr = 0.001
    generator_lr = 0.0002
    
    # Train and evaluate for different splits
    for split_ratio in [0.2, 0.1]:
        print(f"\nTraining with {split_ratio*100}% test split")
        test_loader = get_dataloaders(split_ratio)
        
        # Train small student
        print("\nTraining Small Student")
        opt_small = optim.Adam(small_student.parameters(), lr=student_lr)
        opt_gen_small = optim.Adam(generator.parameters(), lr=generator_lr)
        
        for epoch in tqdm(range(num_epochs)):
            train_step(teacher, small_student, generator, BATCH_SIZE, 
                      (opt_small, opt_gen_small))
        
        # Evaluate small student
        accuracy, conf_matrix = evaluate(small_student, test_loader)
        with torch.no_grad():
            generated = generator(torch.randn(25, LATENT_DIM).to(device))
        plot_results(f'small_student_{split_ratio}', accuracy, conf_matrix, generated)
        
        # Train medium student
        print("\nTraining Medium Student")
        opt_medium = optim.Adam(medium_student.parameters(), lr=student_lr)
        opt_gen_medium = optim.Adam(generator.parameters(), lr=generator_lr)
        
        for epoch in tqdm(range(num_epochs)):
            train_step(teacher, medium_student, generator, BATCH_SIZE,
                      (opt_medium, opt_gen_medium))
        
        # Evaluate medium student
        accuracy, conf_matrix = evaluate(medium_student, test_loader)
        with torch.no_grad():
            generated = generator(torch.randn(25, LATENT_DIM).to(device))
        plot_results(f'medium_student_{split_ratio}', accuracy, conf_matrix, generated)

if __name__ == "__main__":
    # Ensure teacher model weights are downloaded before starting
    main() 