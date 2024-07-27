import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict
import numpy as np

path_to_model = 'DenseNet/model/Trained_Models/Training_A.pth'
path_to_image = 'DenseNet/Images/0/47784381729225956156725553701446715500_sgkre8.png'

chosen_model = "DenseNet201"  # "DenseNet201" or "DenseNet169"

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

net = load_model(path_to_model, net)

testTransform = transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def show_image(image_tensor):
    # Convert the tensor to a PIL image
    unloader = transforms.ToPILImage()
    image = unloader(image_tensor)
    image.show()
    
arr = np.array(Image.open(path_to_image))
arr = (arr - arr.min()) * 255 // (arr.max() - arr.min() + 1e-10)
image = Image.fromarray(arr.astype("uint8"))
#image = Image.open('Images/0/246312850064830034880733297114561684862_yrq3td.png')

image = testTransform(image).unsqueeze(0)

#show_image(image)

with torch.no_grad():
    output = net(image)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()  
    class_names = ['NO', 'YES']  # NO: 'There is no electrical device', YES: 'There is an electrical device'
    print("Is there an electrical device in the image? ",class_names[int(preds.item())])


