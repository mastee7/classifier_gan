from model import ResNet18, weights_init  # Import ResNet18 and weight initialization
from train import train_cnn
from utils import plot_losses, plot_accuracies, save_checkpoint, evaluate_model
from config_mod import torch, num_epochs, device, testloader
import torch.optim as optim
import os
import random

def main():
    # Setting random seed for reproducible training
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Train the model
    resnet18, optimizer, training_losses, validation_losses, training_accuracies, validation_accuracies = train_cnn(num_epochs)

    # Save model
    checkpoint_dir = "/home/wchung25/eee515/HW2/EEE515_HW2/checkpoint_res_cnn"
    
    # Ensure the directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "resnet_checkpoint.pth")
    
    # Save plot
    plot_filename = os.path.join(checkpoint_dir, "resnet_training_validation_loss_plot.png")
    plot_losses(training_losses, validation_losses, plot_filename)
    
    accuracy_plot_filename = os.path.join(checkpoint_dir, "resnet_training_validation_accuracy_plot.png")
    plot_accuracies(training_accuracies, validation_accuracies, accuracy_plot_filename)

    # Save the model checkpoint
    save_checkpoint(resnet18, optimizer, filename=checkpoint_path)

    # Evaluate the trained model on the test data
    evaluate_model(resnet18, testloader, device)
    
if __name__ == '__main__':
    main()
