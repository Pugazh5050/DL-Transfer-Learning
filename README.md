# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
The objective of this experiment is to classify images into their respective categories using a deep neural network. Instead of training a convolutional neural network from scratch, the pretrained VGG19 model is used as a feature extractor. Transfer learning is applied by using the knowledge learned by VGG19 from the ImageNet dataset and adapting its final classification layer to the classes in the given dataset.


## Neural Network Model
<img width="650" height="600" alt="image" src="https://github.com/user-attachments/assets/0ad68eac-47a5-4bbf-9b8d-83072239ff08" />

## DESIGN STEPS
### STEP 1: 

Load the image dataset and divide it into training, validation, and testing sets.

### STEP 2: 

Resize all images to 224 × 224 pixels and preprocess them using the VGG19 preprocessing function.

### STEP 3: 

Load the pretrained VGG19 model with ImageNet weights and remove its original classification layer.

### STEP 4: 

Freeze the pretrained convolutional layers and add new fully connected layers suitable for the dataset.

### STEP 5: 

Compile the model using the Adam optimizer and sparse categorical cross-entropy loss function, then train the model.

### STEP 6: 


Evaluate the trained model using the test dataset and generate the confusion matrix, classification report, and predictions for new sample images.


## PROGRAM

### Name: Pugazhalenthi V

### Register Number:212224100047

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset_path=r"C:\Users\admin\Downloads\chip_data\dataset"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()
show_sample_images(train_dataset)
print(f"Total number of training samples: {len(train_dataset)}")
print(f"Total number of testing samples: {len(test_dataset)}")
first_image, first_label = train_dataset[0]
print(f"First image shape: {first_image.shape}")
first_image1, label = test_dataset[0]
print(f"Shape of the first image : {first_image.shape}")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss' )
    plt.legend()
    plt.show()
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
train_model(model, train_loader,test_loader)
test_model(model, test_loader)
def predict_image(model, image_index, dataset):
    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

    predicted = torch.argmax(output, dim=1).item()

    class_names = dataset.classes

    image_to_display = transforms.ToPILImage()(image)

    plt.figure(figsize=(5, 5))
    plt.imshow(image_to_display)
    plt.title(
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted]}",
        fontsize=16
    )
    plt.axis("off")
    plt.show()

    print(f"Actual: {class_names[label]}, Predicted: {class_names[predicted]}")
predict_image(model, image_index=25, dataset=test_dataset)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="1034" height="796" alt="image" src="https://github.com/user-attachments/assets/937d94f9-5e17-48d0-8d19-5016c2f5b9a1" />


## Confusion Matrix

<img width="969" height="828" alt="image" src="https://github.com/user-attachments/assets/b2f2cd67-a3eb-474c-927c-66317964ccaa" />


## Classification Report
<img width="702" height="278" alt="image" src="https://github.com/user-attachments/assets/741af35d-2bc3-482a-9903-5284f3f379bc" />


### New Sample Data Prediction
<img width="617" height="684" alt="image" src="https://github.com/user-attachments/assets/90a07581-3338-41f5-aa18-5a6ab260e870" />


## RESULT
Thus, an image classification model was successfully developed using transfer learning with the VGG19 architecture. The pretrained VGG19 model was used for feature extraction, and the modified classification layer accurately classified images in the given dataset. The performance of the model was evaluated using loss curves, confusion matrix, classification report, and sample image prediction.

I prefer this response
