import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # Added numpy for safety

# --- 1. SET UP TRANSFORMATIONS ---
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- 2. LOAD YOUR TEST DATA ---
test_dir = './test' 

if not os.path.isdir(test_dir):
    print(f"Error: Test directory '{test_dir}' not found.")
    exit()

test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
print(f"Loaded {len(test_dataset)} images from the test set.")

class_names = test_dataset.classes
print(f"Classes being tested: {class_names}")

# --- 3. REBUILD AND LOAD THE TRAINED MODEL ---
num_classes = len(class_names) 

model = models.mobilenet_v3_small(weights=None)
last_layer_in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features=last_layer_in_features, out_features=num_classes)

# Fixed the warning by adding weights_only=True (optional but good practice)
model.load_state_dict(torch.load('lettuce_detector.pth', map_location=torch.device('cpu'))) # removed weights_only arg if on older pytorch, strictly it should be default safe but explicit helps. 
# If you get an error about weights_only, just remove that argument.
print("Trained model weights loaded successfully.")
model.eval()

# --- 4. EVALUATE THE MODEL ON THE TEST SET ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nEvaluating on device: {device}...")

correct = 0
total = 0

# --- NEW: Initialize lists to store all labels and predictions for the Matrix ---
y_true = []
y_pred = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # --- NEW: Collect the data for the Confusion Matrix ---
        # We must move them to CPU and convert to numpy for sklearn
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        # --- Print predictions for each batch ---
        print(f"\n--- Batch {i+1} Predictions ---")
        for j in range(inputs.size()[0]):
            predicted_class = class_names[predicted[j]]
            actual_class = class_names[labels[j]]
            result_symbol = "✅" if predicted_class == actual_class else "❌"
            print(f"  {result_symbol} Predicted: '{predicted_class}', Actual: '{actual_class}'")


# Calculate and print the final accuracy
accuracy = 100 * correct / total
print("\n--- Final Test Result ---")
print(f'Accuracy on the test set: {accuracy:.2f}%')
print("\nNote: ✅ indicates a correct prediction, ❌ indicates an incorrect prediction.")

# --- 5. GENERATE CONFUSION MATRIX ---
print("Generating Confusion Matrix...")

# Now y_true and y_pred actually exist!
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10)) # Made figure slightly larger for readability
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
plt.xticks(rotation=45, ha='right') # Rotate labels so they don't overlap
plt.tight_layout() # Ensures labels aren't cut off when saving
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.show()