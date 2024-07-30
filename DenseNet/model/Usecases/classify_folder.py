import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

path_to_model = 'DenseNet/model/Trained_Models/Training_C.pth'
path_to_folder = 'DenseNet/Images/1/'

chosen_model = "DenseNet201"  # "DenseNet201" or "DenseNet169"

# Choose the architecture
if chosen_model == "DenseNet169":
    net = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
    net.classifier = nn.Linear(net.classifier.in_features, 1)    
elif chosen_model == "DenseNet201":
    net = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    net.classifier = nn.Linear(net.classifier.in_features, 1)


def load_model(path_to_model, net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    used_model = torch.load(path_to_model, map_location=device)
    net.load_state_dict(used_model['state_dict'])
    net.to(device)
    net.eval()
    return net

# Load the model
net = load_model(path_to_model, net)

testTransform = transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# Optional: display the image
def show_image(image_tensor):
    # Convert the tensor to a PIL image
    unloader = transforms.ToPILImage()
    image = unloader(image_tensor)
    image.show()
    
def process_image(path_to_image):
    arr = np.array(Image.open(path_to_image))
    arr = (arr - arr.min()) * 255 // (arr.max() - arr.min() + 1e-10)
    image = Image.fromarray(arr.astype("uint8"))
    image = testTransform(image).unsqueeze(0)
    return image

def predict_image(net, image):
    with torch.no_grad():
        output = net(image)
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).int()  
        class_names = ['NO', 'YES']  # NO: 'There is no electrical device', YES: 'There is an electrical device'
        return class_names[int(preds.item())]

for image_file in os.listdir(path_to_folder):
    image_path = os.path.join(path_to_folder, image_file)
    # Check if the file is an image
    if image_path.endswith(('.png', '.jpg', '.jpeg')): 
        # Try to open the image
        try:
            # Normalize and transform the image
            image_tensor = process_image(image_path)
            # Predict the image class
            prediction = predict_image(net, image_tensor)
            print(f"Is there a foreign object in the image {image_file}? {prediction}")
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
