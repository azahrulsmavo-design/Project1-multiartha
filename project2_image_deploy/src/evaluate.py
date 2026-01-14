import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import os
from model import SimpleCNN

BATCH_SIZE = 64
MODEL_PATH = "models/cifar10_cnn.pt"
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def evaluate():
    print("Setting up device and data for evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Please train first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    print("Running inference on test set...")
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    print("\nGenerating confusion matrix plot...")
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CIFAR-10 Confusion Matrix')
    
    if not os.path.exists('reports'):
        os.makedirs('reports')
    plt.savefig('reports/cifar10_confusion_matrix.png')
    print("Plot saved to reports/cifar10_confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
