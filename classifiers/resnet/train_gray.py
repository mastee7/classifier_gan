from model import ResNet18, weights_init, nn
from config_gray import torch, device, trainloader, validloader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau   
from sklearn.metrics import confusion_matrix
import numpy as np

# Send model to CUDA
resnet18 = ResNet18().to(device)
    
# Apply weights
resnet18.apply(weights_init)

# Set optimizers and criterion
# Defines the loss function as cross-entropy loss, commonly used in classification tasks
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.05, momentum=0.9)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min')

def train_cnn(num_epochs):
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        resnet18.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet18(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_training_loss = running_loss / len(trainloader)
        training_losses.append(avg_training_loss)
        train_accuracy = 100 * train_correct / train_total
        training_accuracies.append(train_accuracy)

        # Validation phase
        resnet18.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for data in validloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet18(inputs).squeeze()
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        avg_validation_loss = valid_loss / len(validloader)
        validation_losses.append(avg_validation_loss)
        valid_accuracy = 100 * valid_correct / valid_total
        validation_accuracies.append(valid_accuracy)

        scheduler.step(avg_validation_loss)

        print(f'Epoch {epoch + 1} - Train Loss: {avg_training_loss:.3f}, Train Acc: {train_accuracy:.2f}%, Valid Loss: {avg_validation_loss:.3f}, Valid Acc: {valid_accuracy:.2f}%')

    print('Finished Training')
    return resnet18, optimizer, training_losses, validation_losses, training_accuracies, validation_accuracies

def test_cnn(model, testloader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Append predictions and labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(testloader)
    test_accuracy = 100 * test_correct / test_total
    print(f'Test Loss: {avg_test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%')

    # Generating confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizing the confusion matrix

    return avg_test_loss, test_accuracy, cm_normalized
