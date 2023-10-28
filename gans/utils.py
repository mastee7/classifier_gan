import torch
import matplotlib.pyplot as plt
import numpy as np
from train import vutils
from config import device
from data_loader import dataloader

def plot_losses(G_losses, D_losses, filename):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    
def plot_images(img_list, filename):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig(filename)
    plt.close()


def save_checkpoint(generator, discriminator, optimizerG, optimizerD, filename):
    """
    Save a checkpoint for GAN which includes Generator and Discriminator models 
    along with their optimizers.

    Arguments:
    - generator : The Generator model.
    - discriminator : The Discriminator model.
    - optimizerG : The optimizer for the Generator.
    - optimizerD : The optimizer for the Discriminator.
    - filename : The path to save the checkpoint.
    """
    state = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict()
    }
    torch.save(state, filename)


