import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Labels used for training A, these are all electrical device categories
foreign_objects_electrical_device = [
    "electrical device",
    "dual chamber device",
    "single chamber device",
    "pacemaker",
    "dai"
]

# Labels used for training B and C, these are all foreign object categories
foreign_objects = [
    "tracheostomy tube", 
    "endotracheal tube",
    "NSG tube",
    "chest drain tube",
    "ventriculoperitoneal drain tube",
    "gastrostomy tube",
    "nephrostomy tube",
    "double J stent",
    "catheter",
    "central venous catheter",
    "central venous catheter via subclavian vein",
    "central venous catheter via jugular vein",
    "reservoir central venous catheter",
    "central venous catheter via umbilical vein",
    "electrical device",
    "dual chamber device",
    "single chamber device",
    "pacemaker",
    "dai", 
    "artificial heart valve",
    "artificial mitral heart valve",
    "artificial aortic heart valve",
    "metal",
    "osteosynthesis material",
    "sternotomy",
    "suture material",
    "bone cement",
    "prosthesis",
    "humeral prosthesis",
    "mammary prosthesis",
    "endoprosthesis",
    "aortic endoprosthesis",
    "abnormal foreign body",
    "external foreign body"
]

# Will be called if there is no csv file with the image classifications
def load_and_classify_images(labels_file):
    # Generate a dataframe from the csv file
    df = pd.read_csv(labels_file, dtype={19: str, 20: str}, low_memory=False)
    classifications = []
    # Iterate over the rows of the dataframe and save the ID of the image, the folder directory and the labels 
    for index, row in df.iterrows():
        object_row = row[0]
        filename = row['ImageID']
        directory = row['ImageDir']
        csv_labels = row['Labels']
        contained_foreign_objects = []
        
        category = 0
        try:
            for foreign_object_type in foreign_objects:
                # Check if at least one foreign object is in the labels of the image
                if foreign_object_type in csv_labels:
                    # If a foreign object is in the labels, the category is set to 1
                    category = 1
                    # Save the foreign object type in the list of contained foreign objects
                    contained_foreign_objects.append(foreign_object_type)
                    
        # We have NAN objects in the dataset that we can ignore
        except Exception as e:
            print("error in: ", row[0])
            continue 
        
        # Save each filename/image with its folder directory, classification and contained foreign objects in a dictionary
        classifications.append({
            'object': object_row,
            'filename': filename,
            'directory': directory,
            'category': category,
            'contained_foreign_objects': contained_foreign_objects
        })
        
        # Print the progress of the classification
        if index % 1000 == 0:
            print(f"Processed {index} rows.")
            
    # Generate a dataframe from the classifications dictionary
    result_df = pd.DataFrame(classifications)
        
    return result_df 

# Get all forein object names
def get_foreign_objects():
    return foreign_objects  
  
# Split the data into a training and a test set
def split_data(df, split_ratio):
    train_df, test_df = train_test_split(df, test_size=(1 - split_ratio))
    return train_df, test_df
    
# This function checks if the classified images csv exists and loads it if it does,
# otherwise it classifies the images and generates the csv
def check_if_classified_images_csv_exists(classified_images_csv, labels_file):
    if os.path.exists(classified_images_csv):
        classified_images_df = pd.read_csv(classified_images_csv)
    else:
        classified_images_df = load_and_classify_images(labels_file)
        classified_images_df.to_csv(classified_images_csv, index=False)
    return classified_images_df
