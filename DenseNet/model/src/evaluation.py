import torch
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
import ast
import matplotlib.pyplot as plt

import main
import classify_images

# Generates a DataFrame with counters for each foreign object 
# to evaluate the performance depending on the object type
def object_categories_initialization():
    object_categories = classify_images.get_foreign_objects()
    data = []

    # Loop through each object in the list
    for obj in object_categories:
        # Create a dictionary for each object 
        obj_data = {
            'object_name': obj,
            'total_amount': 0,  
            'correct': 0        
        }
        # Append the dictionary to the data list
        data.append(obj_data)

    # Create a DataFrame from the data list and return it
    objects_df = pd.DataFrame(data)
    return objects_df
    
def get_main_df():
    return main.get_classified_images()

def evaluate_object_categories(labels, preds, img_name, objects_df, main_df):

    if len(labels) != len(preds) or len(labels) != len(img_name):
        print("Error: Length of labels and predictions do not match.")
        return
    
    for i in range(len(labels)):
        # For each image labelled as containing a foreign object update counters
        if labels[i] == 1:
            obj_list = main_df.loc[main_df['filename'] == img_name[i], 'contained_foreign_objects'].values[0]
            obj_list = ast.literal_eval(obj_list)
            
            for obj in obj_list:
                if preds[i] == 1:
                    # If the prediction is correct, update the correct counter and the total amount counter
                    objects_df.loc[objects_df['object_name'] == obj, ['total_amount', 'correct']] += 1
                else:
                    # If the prediction is wrong, update only the total amount counter
                    objects_df.loc[objects_df['object_name'] == obj, 'total_amount'] += 1
    return objects_df

# Evaluate the model
def evaluate(model, criterion, test_loader, device, logger):
    logger.info("Evaluation started...")
    # Set the model to evaluation mode
    model.eval() 
    
    running_loss = 0.0
    processed_data = 0
    all_labels = []
    all_preds = []
    all_probs = []
    amt_images = len(test_loader.dataset)
    # Initialize the object categories DataFrame
    objects_df = object_categories_initialization()
    
    # Get the DataFrame with the classified images in order to get the labels
    main_df = get_main_df()
    
    # Process the test data in batches
    with torch.no_grad():
        for inputs, labels, img_name in test_loader:
            
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            # Generate probabilities for the images
            outputs = model(inputs)
            
            # Calculate the loss based on the probabilities 
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)

            # Convert probabilities to binary predictions
            preds = (probs > 0.6).int() 
            
            # Add the results to the object categories DataFrame and update its counters
            objects_df = evaluate_object_categories(labels, preds, img_name, objects_df, main_df)
            
            # Calculate the loss of the batch
            loss = loss.item()
            
            # Calculate the total loss so far and the number of processed images
            running_loss += loss * inputs.size(0)
            processed_data += inputs.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    total_loss = running_loss / amt_images
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    logger.info(f" -  Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    logger.info(f" -  AUC: {auc:.4f}")  
    logger.info(f" -  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")  
    logger.info(f" -  Total: {tp + fp + fn + tn}, Correct: {tp + tn}, Wrong: {fp + fn}")  
    logger.info("Detailed object category results:")
    logger.info(f'\n{objects_df.to_string()}')
    logger.info("Evaluation finished.")
    logger.info("----------------------------------------------------------------------------------- ")
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

