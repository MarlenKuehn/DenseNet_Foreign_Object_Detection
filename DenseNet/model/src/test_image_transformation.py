import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from skimage import io
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch

# Load an image

# Check the shape of the image
#print("Image shape:", img.shape)

# Display the image with the correct colormap for grayscale
image = Image.open('Images/0/47784381729225956156725553701446715500_sgkre8.png')

#image.show()

#bad

arr = np.array(Image.open("Images/0/7229642368407186565500544115495475569_ja0kjg.png"))
arr = (arr - arr.min()) * 255 // (arr.max() - arr.min() + 1e-10)
img = Image.fromarray(arr.astype("uint8"))
img.show()

#good


normMean = [0.49139968]
normStd = [0.24703233]
normTransform = transforms.Normalize(normMean, normStd)

trainTransform = transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.RandomAdjustSharpness(2, p=0.5),
    transforms.ToTensor(),
    #normTransform
])

transformed_image = trainTransform(img)

def show_image(image_tensor):
    # Convert the tensor to a PIL image
    unloader = transforms.ToPILImage()
    image = unloader(image_tensor)
    image.show()

show_image(transformed_image)

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# Denormalize the transformed image tensor
denorm_image_tensor = denormalize(transformed_image.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

# Convert the tensor to a numpy array for visualization
transformed_image_array = denorm_image_tensor.numpy().transpose((1, 2, 0))

# Clip the values to be in the valid range [0, 1]
transformed_image_array = np.clip(transformed_image_array, 0, 1)

# Display the original and transformed images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

# Transformed image
axs[1].imshow(transformed_image_array)
axs[1].set_title("Transformed Image")
axs[1].axis('off')

#plt.show()