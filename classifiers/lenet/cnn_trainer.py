from model import *
from config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau   # to reduce the learning rate as we go

# Send model to CUDA
net_cnn = Net().to(device)

# Apply weights
net_cnn.apply(weights_init)

# Set optimizers and criterion
# Defines the loss function as cross-entropy loss, commonly used in classification tasks
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_cnn.parameters(), lr=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def train_cnn(num_epochs):
    
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # Training phase
        net_cnn.train()
        # Initializ a variable to accumulate loss over an epoch
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net_cnn(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            if i % 2000 == 1999:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] - Loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        avg_training_loss = running_loss / len(trainloader)
        training_losses.append(avg_training_loss)
        train_accuracy = 100 * train_correct / train_total
        training_accuracies.append(train_accuracy)

        # Validation phase
        net_cnn.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for data in validloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net_cnn(inputs).squeeze()
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                
        avg_validation_loss = valid_loss / len(validloader)
        validation_losses.append(avg_validation_loss)
        valid_accuracy = 100 * valid_correct / valid_total
        validation_accuracies.append(valid_accuracy)        
        
        # Adjust based on validation metric
        scheduler.step(valid_loss / valid_total)  

        print(f'Epoch {epoch + 1} - Validation Loss: {valid_loss / valid_total:.3f}')

    print('Finished Training')
    return net_cnn, optimizer, training_losses, validation_losses, training_accuracies, validation_accuracies
