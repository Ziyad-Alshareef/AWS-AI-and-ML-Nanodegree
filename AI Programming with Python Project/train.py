import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Define a function for parsing command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep neural network.")
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg13', 'vgg16'], help='Choose architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    return parser.parse_args()

# Define a function to load the data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

    
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12)
    validloader = DataLoader(valid_data, batch_size=32, shuffle=False)
    testloader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return trainloader, validloader, testloader, train_data

# Define a function to build the model
def build_model(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def validation_loss_accuracy(model, validloader, criterion, gpu):
    model.eval()  # Set model to evaluation mode
    valid_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in validloader:
            if gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return valid_loss / len(validloader), accuracy

# Modify the train_model function to include validation loss and accuracy
def train_model(model, trainloader, validloader, epochs, learning_rate, gpu):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    if gpu and torch.cuda.is_available():
        model = model.cuda()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            if gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Compute the validation loss and accuracy
        valid_loss, accuracy = validation_loss_accuracy(model, validloader, criterion, gpu)

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Training loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss:.3f}.. "
              f"Validation accuracy: {accuracy:.3f}%")
    return optimizer
# Define a function to save the checkpoint

def save_checkpoint(model, optimizer, epochs, save_dir, train_data, lr, arch, HU):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs,
        'lr':lr,
        'classifier': model.classifier,
        'hidden_units': HU
    }
    torch.save(checkpoint, save_dir)
    print(f"Checkpoint saved at {save_dir}")

# Main function
def main():
    args = parse_args()
    trainloader, validloader, testloader,train_data = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    
    opt=train_model(model, trainloader, validloader, args.epochs, args.learning_rate, args.gpu)
    
    save_checkpoint(model, opt, args.epochs, args.save_dir, train_data, args.learning_rate, args.arch, args.hidden_units)

if __name__ == '__main__':
    main()