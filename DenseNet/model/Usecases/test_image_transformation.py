from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision

# Load an image and normailze it
arr = np.array(Image.open("DenseNet/Images/0/7229642368407186565500544115495475569_ja0kjg.png"))
arr = (arr - arr.min()) * 255 // (arr.max() - arr.min() + 1e-10)
img = Image.fromarray(arr.astype("uint8"))

# Show the original image
img.show()

trainTransform = transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.RandomAdjustSharpness(2, p=0.5),
    transforms.ToTensor(),
])

# Transform the image
transformed_image = trainTransform(img)

def show_image(image_tensor):
    # Convert the tensor to a PIL image
    unloader = transforms.ToPILImage()
    image = unloader(image_tensor)
    image.show()
    
# Display the transformed image
show_image(transformed_image)
