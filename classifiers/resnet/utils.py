import torch
import matplotlib.pyplot as plt
import seaborn as sns
from config_skewed import id_to_class

def plot_accuracies(training_accuracies, validation_accuracies, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracies Over Epochs')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_losses(training_losses, validation_losses, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    
def plot_confussion(cm_normalized, filename):
    # Plotting the confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=id_to_class.values(), 
                yticklabels=id_to_class.values())
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def save_checkpoint(model, optimizer, filename):
    """
    Save a model checkpoint.
    
    Arguments:
    - model : The neural network model.
    - optimizer : The optimizer.
    - filename : The path to save the checkpoint.
    """
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)
    
    
def evaluate_model(model, dataloader, device):
    model.eval()  # set the model to evaluation mode
    
    correct = 0
    total = 0

    with torch.no_grad():  # we don't need gradients for evaluation
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # get the predicted class for each sample
            
            total += labels.size(0)  # increase the total count
            correct += (predicted == labels).sum().item()  # increase the correct count if prediction is correct

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy:.2f}%')

    return accuracy

