import torch
from torchvision import models
import torch.nn as nn
import os

# --- 1. PRE-LOAD AND INSPECT THE SAVED MODEL ---
# This is the most robust way to ensure the model structure matches.

pth_file = 'lettuce_detector.pth'
if not os.path.exists(pth_file):
    print(f"Error: Trained model file '{pth_file}' not found.")
    print("Please make sure your trained model is in the same folder as this script.")
    exit()

print(f"Inspecting saved model file: '{pth_file}'...")
# Load the state dictionary from your saved file first.
# We use weights_only=True for security, which is the modern standard.
state_dict = torch.load(pth_file, map_location=torch.device('cpu'), weights_only=True)

# Dynamically determine the number of classes from the shape of the classifier's weight tensor.
# The key for the final layer's weights in MobileNetV3 is 'classifier.3.weight'.
try:
    num_classes = state_dict['classifier.3.weight'].shape[0]
    print(f"Detected {num_classes} classes from the saved model.")
except KeyError:
    print("Error: Could not find the key 'classifier.3.weight' in the saved model.")
    print("This might mean the model saved is not a MobileNetV3. Please check your training script.")
    exit()

# --- 2. DEFINE THE MODEL ARCHITECTURE TO MATCH THE SAVED FILE ---

print("Rebuilding the MobileNetV3 model structure...")
# Load the MobileNetV3 model structure (without pre-trained weights)
model = models.mobilenet_v3_small(weights=None) 
# Get the number of input features from the model's last layer
last_layer_in_features = model.classifier[-1].in_features
# Now, create the final layer with the correct number of input and output features
model.classifier[-1] = nn.Linear(in_features=last_layer_in_features, out_features=num_classes)
print("Model structure is ready.")

# --- 3. LOAD YOUR TRAINED WEIGHTS ---
print(f"Loading trained weights from '{pth_file}'...")
# Now that the model structure is a perfect match, this will work.
model.load_state_dict(state_dict)
print("Weights loaded successfully.")

# Set the model to evaluation mode (this is important for conversion)
model.eval()

# --- 4. CONVERT THE MODEL TO ONNX ---
# Create a dummy input tensor with the correct shape (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224) 
onnx_output_path = "lettuce_detector.onnx"

print(f"Exporting model to ONNX format at '{onnx_output_path}'...")
torch.onnx.export(model,
                  dummy_input,
                  onnx_output_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],   # The name for the input tensor
                  output_names=['output'], # The name for the output tensor
                  dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

print("\n--- Conversion Complete! ---")
print(f"Your model has been saved as '{onnx_output_path}'.")
print("You can now use this file with the HTML interface or on your Raspberry Pi.")

