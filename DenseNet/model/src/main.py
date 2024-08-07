import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import models, transforms

import pandas as pd

import classify_images
import process_images
import training
import evaluation
import log

# Settings:
chosen_model = "DenseNet201"  # "DenseNet201" or "DenseNet169"
split_ratio = 0.8 # 80% training, 20% testing 
learning_rate = 0.0001 # learning rate lr for the optimizer, can be adjusted, smaller is more precise, bigger is faster
num_epochs = 1 # rounds through the entire dataset
fraction_to_keep = 0.13 # 0.025 used for electrical devices
batch_size = 32 # number of samples per batch to load

model_name = "foreign_objects_1" # Name of the model for the savings
log_file = "DenseNet/model/Trained_Models/log_file_1.txt" # Path to the written log file
labels_file = "DenseNet/model/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv" # Path to the PadChest csv file with the official labels
images_folder = "DenseNet/Images" # Path to the folder with the images
classified_images_csv = "DenseNet/model/classify_images.csv" # Path to the potential generated csv file with the classified images

# Check if the classified images csv exists, otherwise generate it
classified_images_df = classify_images.check_if_classified_images_csv_exists(classified_images_csv, labels_file)

# Return the classified images DataFrame
def get_classified_images():
    return classified_images_df

def main():
     # Split the dataset into training and testing
    train_df, test_df = classify_images.split_data(classified_images_df, split_ratio=split_ratio)

    # Balance the training dataset by reducing the number of 0-classified images
    zero_classified = train_df[train_df['category'] == 0]
    one_classified = train_df[train_df['category'] == 1]
    zero_classified_to_keep = zero_classified.sample(frac=fraction_to_keep, random_state=42)
    train_df = pd.concat([one_classified, zero_classified_to_keep]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Transformations on the training dataset
    trainTransform = transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.RandomAdjustSharpness(2, p=0.5),
        transforms.ToTensor(),
    ])
    
    # Transformations on the testing dataset
    testTransform = transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.49139968], [0.24703233])
    ])
    
    # Set up the logger 
    logger = log.setup_logging(log_file)
    
    logger.info("----------------------------------------------------------------------------------- ")
    logger.info(f"Model: {chosen_model} // Epochs: {num_epochs}")
    logger.info(f"Split Ratio: {split_ratio} // Learning Rate: {learning_rate} // Batch Size: {batch_size} // Fraction of neg. Samples: {fraction_to_keep}")
    logger.info("----------------------------------------------------------------------------------- ")
    
    # Define training dataset and DataLoader
    train_dataset = process_images.ImageDataset(df=train_df, image_directory=images_folder, logger=logger, transform=trainTransform, name='Training')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=process_images.collate_fn)
    
    # Define testing dataset and DataLoader
    test_dataset = process_images.ImageDataset(df=test_df, image_directory=images_folder, logger=logger, transform=testTransform, name='Testing')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=process_images.collate_fn)
    logger.info("----------------------------------------------------------------------------------- ")
    
    # Select and define the chosen model architecture, weights are the pre-trained weights on the ImageNet dataset
    # The last layer called classifier layer is replaced by a linear layer with one output for binary classification
    if chosen_model == "DenseNet169":
        net = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        net.classifier = nn.Linear(net.classifier.in_features, 1)    
    elif chosen_model == "DenseNet201":
        net = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        net.classifier = nn.Linear(net.classifier.in_features, 1)
    
    # Optimize the learning of the model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Define the optimizer and loss function
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    training.train(net, criterion, optimizer, train_loader, num_epochs, device, model_name, logger)
    
    # Evaluate the model
    evaluation.evaluate(net, criterion, test_loader, device, logger)

    # Close the logger
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
if __name__=='__main__':
    main()
