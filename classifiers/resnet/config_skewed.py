import torch 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

class CIFAR10S(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, skew_ratio=0.95, all_grayscale=False):
        super().__init__(root, train, transform, target_transform, download)
        self.skew_ratio = skew_ratio
        self.all_grayscale = all_grayscale
        self._skew_dataset()
        
    def _convert_to_grayscale(self, indices):
        for idx in indices:
            gray_image = np.dot(self.data[idx][...,:3], [0.2989, 0.5870, 0.1140])
            self.data[idx] = np.repeat(gray_image[:, :, np.newaxis], 3, axis=2).astype(self.data[idx].dtype)

    def _skew_dataset(self):
        np.random.seed(42)  # For reproducibility
        for class_label in range(10):  # CIFAR-10 has 10 classes
            class_indices = np.where(np.array(self.targets) == class_label)[0]
            np.random.shuffle(class_indices)
            
            # Apply skewing for training data
            if self.train:
                split_point = int(len(class_indices) * self.skew_ratio)
                if class_label < 5:  # First 5 classes are 95% color, 5% grayscale
                    grayscale_indices = class_indices[split_point:]
                else:  # Next 5 classes are 95% grayscale, 5% color
                    grayscale_indices = class_indices[:split_point]
                self._convert_to_grayscale(grayscale_indices)
            # Convert the entire test dataset to grayscale if required
            elif self.all_grayscale:
                self._convert_to_grayscale(class_indices)


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

# Load the train dataset
trainset = CIFAR10S(root='./data', train=True, download=True, transform=augmented_transform)

# Create two separate test sets
testset_color = CIFAR10S(root='./data', train=False, download=True, transform=transform, all_grayscale=False)
testset_gray = CIFAR10S(root='./data', train=False, download=True, transform=transform, all_grayscale=True)

# Splitting the trainset into train and validation sets
train_size = int(0.8 * len(trainset))  # 80% of the dataset
valid_size = len(trainset) - train_size  # Remaining 20% for validation

train_dataset, valid_dataset = random_split(trainset, [train_size, valid_size])

# Train and Validation DataLoaders
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Test DataLoaders
testloader_color = DataLoader(testset_color, batch_size=batch_size, shuffle=False, num_workers=2)
testloader_gray = DataLoader(testset_gray, batch_size=batch_size, shuffle=False, num_workers=2)


# Class label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

id_to_class = {
    0: "plane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
