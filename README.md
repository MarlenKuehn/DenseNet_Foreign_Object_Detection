<!-- markdownlint-disable -->

# DenseNet_Foreign_Object_Detection
This framework uses a DenseNet architecture to detect Foreign Objects in chest x-ray images.

## Introduction



## Installation Manual

1. In order to be able to run the framework you need a Python environment. The used Python version for the implementation was 3.11.4. If necessary install Python from https://www.python.org/downloads/.

2. Make sure you have all the necessary libraries installed. You can install them by running the following command:

```
pip install -r requirements.txt
```

or run the commands for the packages you are missing individually. The required packages are:

```
pip install matplotlib==3.8.0
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install Pillow==10.4.0
pip install scikit_learn==1.3.2
pip install skimage==0.0
pip install torch==2.3.0
pip install torchvision==0.18.0
```

3. Clone this repository: ```git clone https://github.com/MarlenKuehn/DenseNet_Foreign_Object_Detection.git``` and load it into a local folder directory.

4. To download the PadChest dataset go to https://bimcv.cipf.es/bimcv-projects/padchest/ and click on 'Download complete dataset'. You need to fill out the form and agree to the terms of use. After that you will recieve the link to the download page of the dataset. Download the folders 1-50 and 54. Keep the original folder structure and names as it is, unzip them and load them into this repository inside a folder called Images in the directory DenseNet/Images/. 
   
5. To verify that the dataset is correctly loaded you can use the 'Verify_Zips_ImageCounts.csv.xlsx' from the download page. It contains the number of images in each file. 

6. Download from the same page the file 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv', it contains the original labels for the images. Put it into the directory 'DenseNet/model/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'.
   
7. To access the pretained models go to https://drive.google.com/drive/folders/18dPp1yi6RvYIGMRyjZoqxCColMbDsWM9?usp=sharing, download the files 'Training_A.pth', 'Training_B.pth' and 'Training_C.pth' or just the model of your choice and put them into the directory 'DenseNet/model/Trained_Models/'.

8. Now you should be able to run main.py. The first time you run it, it will generate the classify_images.csv file, which classifies the images into the classes 1 for containing a foreign object and 0 for not containing a foreign object.
   
## Further Uscases

In the 'DenseNet/model/Usecases' folder are further usecases implemented:

- The 'test_image_transformation.py' can be used to visualize different transformations done to the images. 
  
- You can add an image directory and a chosen model into the classify_images.py file to classify whether the image contain a foreign object or not. In order to do this for a folder filled with images, you can use the 'classify_folder.py' file accordingly.