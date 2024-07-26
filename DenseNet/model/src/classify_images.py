import pandas as pd
import os
from sklearn.model_selection import train_test_split


foreign_objects_electrical_device = [
    "electrical device",
    "dual chamber device",
    "single chamber device",
    "pacemaker",
    "dai"
]

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

def load_and_classify_images(labels_file):
          
    df = pd.read_csv(labels_file, dtype={19: str, 20: str}, low_memory=False)
    classifications = []
    
    for index, row in df.iterrows():
        object_row = row[0]
        filename = row['ImageID']
        directory = row['ImageDir']
        csv_labels = row['Labels']
        contained_foreign_objects = []
        
        category = 0
        try:
            for foreign_object_type in foreign_objects:
                if foreign_object_type in csv_labels:
                    category = 1
                    contained_foreign_objects.append(foreign_object_type)
        except Exception as e:
            print("error in: ", row[0])
            continue # we have nan objects in the dataset eg object 40181
        
        classifications.append({
            'object': object_row,
            'filename': filename,
            'directory': directory,
            'category': category,
            'contained_foreign_objects': contained_foreign_objects
        })
        
        if index % 1000 == 0:
            print(f"Processed {index} rows.")
            
    result_df = pd.DataFrame(classifications)
        
    return result_df 
  
def get_foreign_objects():
    return foreign_objects  
  
def split_data(df, split_ratio):
    # A split_ratio of 0.8 will allocate 80% of the data to training and 20 % to testing
    train_df, test_df = train_test_split(df, test_size=(1 - split_ratio))
    return train_df, test_df
    
  
def check_if_classified_images_csv_exists(classified_images_csv, labels_file):
    if os.path.exists(classified_images_csv):
        classified_images_df = pd.read_csv(classified_images_csv)
    else:
        classified_images_df = load_and_classify_images(labels_file)
        classified_images_df.to_csv(classified_images_csv, index=False)
    return classified_images_df
