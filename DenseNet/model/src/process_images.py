import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, df, image_directory, logger, transform=None, name=None):
        self.df =  df # pd.read_csv(df) to read instead from a folder
        self.image_directory = image_directory
        self.logger = logger
        self.transform = transform
        self.name = name
        self.filtered_data_frame = self._filter_missing_images()
        self.pos_count, self.neg_count = self._count_labels()
        
        
    def _filter_missing_images(self):
        valid_rows = []
        for idx, row in self.df.iterrows():
            subdir = row['directory']
            img_name = os.path.join(self.image_directory, str(subdir), row[1])
            if os.path.exists(img_name):
                valid_rows.append(row)
            #else:
                #print(f"Warning: Image file '{img_name}' not found.")
                #print("Skipped image.")
        return pd.DataFrame(valid_rows)
    
    def _count_labels(self):
        num_positives = self.filtered_data_frame[self.filtered_data_frame['category'] == 1].shape[0]
        num_negatives = self.filtered_data_frame[self.filtered_data_frame['category'] == 0].shape[0]
        total_samples = num_positives + num_negatives
        self.logger.info(f"{self.name} Dataset: Total samples ({total_samples}), positive samples ({num_positives}), negative samples ({num_negatives})")
        return num_positives, num_negatives

    def __len__(self):
        return len(self.filtered_data_frame)

    def __getitem__(self, idx):
        subdir = self.filtered_data_frame.iloc[idx]['directory']
        img_name = self.filtered_data_frame.iloc[idx]['filename']
        img_path = os.path.join(self.image_directory, str(subdir), img_name)
        if not os.path.exists(img_path):
            #print(f"Warning: Image file '{img_path}' not found.")
            #print("Skipped image.")
            # Optionally, you can skip this entry by returning None or raise an error
            return None
        try:
            arr = np.array(Image.open(img_path))
            arr = (arr - arr.min()) * 255 // (arr.max() - arr.min() + 1e-10)
            image = Image.fromarray(arr.astype("uint8"))
            #image.show()
            #print(f"Loaded image: {img_path}")  # Print the path of the loaded image
            #print(f"Image tensor: {image}")  # Print raw pixel values
        except Exception as e:
            #print(f"Image file could not be opened: {e}")
            return None
        try:
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Image could not be transformed: {e}")
            return None
        label = self.filtered_data_frame.iloc[idx]['category']
        #print(f"Label: {label}, image:", img_path)
        return image, label, img_name
    
    
def collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    # If batch is empty, return empty tensors
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    # Use default collate function
    return torch.utils.data.dataloader.default_collate(batch)
