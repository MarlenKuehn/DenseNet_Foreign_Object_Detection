import torch

from torchvision import transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def show_image(image_tensor):
    # Convert the tensor to a PIL image in order to be able to show it
    unloader = transforms.ToPILImage()
    image = unloader(image_tensor)
    image.show()

# Save the model after each epoch in case of a crash
def save_checkpoint(epoch, model, optimizer, loss, filepath):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filepath)

# THe full training process
def train(model, criterion, optimizer, train_loader, num_epochs, device, model_name, logger):
    
    # Set the model to training mode
    model.train()
    logger.info("Training started...")
    
    # Iterate over the epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        processed_data = 0
        all_labels = []
        all_preds = []
        amt_images = len(train_loader.dataset)

        # Process the training data in batches
        for inputs, labels, img_name in train_loader:

            # Inputs are the images, labels are the ground truth values
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Generate probabilities
            outputs = model(inputs)

            # Calculate loss based on the probabilities
            loss = criterion(outputs, labels)
            
            # Calculate predictions out of the probabilities
            probs = torch.sigmoid(outputs)

            # Convert predictions to binary values
            preds = (probs > 0.5).int()  

            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Calculate the loss of the batch, loss.item() returns the loss as a scalar
            loss = loss.item()
            
            # Calculate total loss so far and the number of processed images
            running_loss += loss * inputs.size(0)
            processed_data += inputs.size(0)
            
            # Keep track of all labels and predictions 
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # Calculate accurancy, precision, recall and F1-score for the batch
            accuracy = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            print(f'Epoch {epoch + 1}/{num_epochs} [{processed_data}/{amt_images}]: Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')

        # Calculate accuracy, precision, recall, F1-score, true positives, false positives, 
        # false negatives and true negatives for the epoch
        epoch_loss = running_loss / amt_images
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

        logger.info(f" -  Epoch {epoch + 1}/{num_epochs}: Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
        logger.info(f" -  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")  
        logger.info(f" -  Total: {tp + fp + fn + tn}, Correct: {tp + tn}, Wrong: {fp + fn}\n")  

        # Save the model after each epoch
        checkpoint_filepath = f'DenseNet/model/Trained_Models/{model_name}_epoch_{epoch+1:02d}.pth'
        save_checkpoint(epoch, model, optimizer, epoch_loss, checkpoint_filepath)
        
    logger.info("Training finished and model saved.")
    logger.info("----------------------------------------------------------------------------------- ")
    
    