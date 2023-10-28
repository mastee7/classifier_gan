from config import image_size, batch_size, workers
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# Define a transform to preprocess the data.
# Adjust as needed.
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Download and load the CelebA dataset
dataset = datasets.ImageFolder(root='/home/wchung25/eee515/HW2/EEE515_HW2/gan/squares', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)