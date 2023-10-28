from train import train_dcgan
import os
from utils_dogs import save_checkpoint, plot_losses, plot_images
from config import torch, num_epochs, device
from train_dogs import train_dcgan

import random

def main():
    # Setting random seed for reproducible training
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    # Train the DCGAN model
    netG, netD, optimizerG, optimizerD, G_losses, D_losses, img_list = train_dcgan(num_epochs, device)  # Adjust parameters as needed

    # Save model
    checkpoint_dir = "/home/wchung25/eee515/HW2/EEE515_HW2/checkpoint_gan_dogs"
    
    # Ensure the directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    # Save loss plot
    plot_filename = os.path.join(checkpoint_dir, "gan_loss_dogs_plot.png")
    plot_losses(G_losses, D_losses, plot_filename)
    
    # Save image plot
    plot_filename = os.path.join(checkpoint_dir, "gan_realvsfake_dogs_plot.png")
    plot_images(img_list, plot_filename)

    save_checkpoint(generator=netG, discriminator=netD, optimizerG=optimizerG, optimizerD=optimizerD, filename=os.path.join(checkpoint_dir, "dcgan_checkpoint.pth"))

    
if __name__ == '__main__':
    main()
