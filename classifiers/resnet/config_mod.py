import torch 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):  # Adjust std as needed
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Normalization specific to CIFAR10 dataset
mean = [0.4914, 0.4822, 0.4465]
std = [0.247, 0.243, 0.261]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)]) 

# The output of torchvision datasets are PILImages or range [0,1]
# Augmented transforms
augmented_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flips the image horizontally
    transforms.RandomRotation(5),      # Rotates the image by up to 5 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translates the image
    transforms.ToTensor(),              # Converts PIL Image to Tensor
    transforms.Normalize(mean, std),    # Normalize the images
    AddGaussianNoise(0., 0.05)           # Adds Gaussian Noise
])

# Batch size
batch_size = 16

# Number of GPUs available. Use 0 for CPU mode
ngpu = 1

device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

num_epochs = 10

# Load the train and test datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=augmented_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Splitting the trainset into train and validation sets
train_size = int(0.8 * len(trainset))  # 80% of the dataset
valid_size = len(trainset) - train_size  # Remaining 20% for validation

train_dataset, valid_dataset = random_split(trainset, [train_size, valid_size])

# Train and Validation DataLoaders
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Test DataLoader
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Class label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
