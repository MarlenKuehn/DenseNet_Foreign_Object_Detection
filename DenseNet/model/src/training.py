
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def show_image(image_tensor):
    # Convert the tensor to a PIL image
    unloader = transforms.ToPILImage()
    image = unloader(image_tensor)
    image.show()
    #plt.show()
    
def save_checkpoint(epoch, model, optimizer, loss, filepath):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filepath)

def train(model, criterion, optimizer, train_loader, num_epochs, device, model_name, logger):
    model.train()
    logger.info("Training started...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        processed_data = 0
        all_labels = []
        all_preds = []
        amt_images = len(train_loader.dataset)

        for inputs, labels, img_name in train_loader:
            #for i in range(min(len(inputs), 3)):  # Display first 5 images
                #img, lbl = inputs[i], inputs[i]
                #show_image(img)
            inputs = inputs.to(device)
            #print("Labels: ", labels)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()

            outputs = model(inputs)
            #print("Outputs: ", outputs)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)

            # Convert probabilities to binary predictions
            preds = (probs > 0.5).int()  
            #print("Preds: ", preds)
            
            # update model
            loss.backward()
            optimizer.step()
            
            loss = loss.item()
            
            running_loss += loss * inputs.size(0)
            processed_data += inputs.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            accuracy = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            print(f'Epoch {epoch + 1}/{num_epochs} [{processed_data}/{amt_images}]: Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')

        epoch_loss = running_loss / amt_images
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

        logger.info(f" -  Epoch {epoch + 1}/{num_epochs}: Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
        logger.info(f" -  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")  
        logger.info(f" -  Total: {tp + fp + fn + tn}, Correct: {tp + tn}, Wrong: {fp + fn}\n")  

        checkpoint_filepath = f'DenseNet/model/Trained_Models/{model_name}_epoch_{epoch+1:02d}.pth'
        save_checkpoint(epoch, model, optimizer, epoch_loss, checkpoint_filepath)
    logger.info("Training finished and model saved.")
    logger.info("----------------------------------------------------------------------------------- ")
    
    