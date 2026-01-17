import os
from torchvision import datasets

# IMPORTANT: This should point to your training directory.
train_dir = './train' 

if not os.path.isdir(train_dir):
    print(f"Error: Directory '{train_dir}' not found.")
    print("Please make sure this script is in the same parent folder as your 'train' directory.")
else:
    # This loads the dataset just to get the class order, exactly as PyTorch sees it.
    train_dataset = datasets.ImageFolder(train_dir)
    class_names = train_dataset.classes
    
    print("\n--- Your Model's True Class Order ---")
    print("Copy the entire array below (from 'const' to ';') and paste it into your index.html file.\n")
    
    # This prints the JavaScript code in the exact format you need.
    print("const CLASS_NAMES = [")
    for name in class_names:
        print(f"    '{name}',")
    print("];")
    print("\n--- End of Class Order ---\n")