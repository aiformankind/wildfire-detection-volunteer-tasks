import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
torch.__version__





if __name__ == '__main__':


    

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'validation':
        transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            normalize
        ]),
    }   

    image_datasets = {
        'train': 
        datasets.ImageFolder(r'C:\Users\sidta\Desktop\AI For Mankind\false_positives_6-latest\wildfire\train', data_transforms['train']),
        'validation': 
        datasets.ImageFolder(r'C:\Users\sidta\Desktop\AI For Mankind\false_positives_6-latest\wildfire\val', data_transforms['validation'])
    }

    print("num classes: " + str(len(image_datasets['train'].classes)))

    dataloaders = {
        'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=32,
                                    shuffle=True, num_workers=2),
        'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                    batch_size=32,
                                    shuffle=False, num_workers=2)
    }
    


    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = models.resnet50(pretrained=True).to(device)
        
    for param in model.parameters():
        param.requires_grad = False   
        
    model.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 26),
                nn.LogSoftmax(dim=1)).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())
    


    def train_model(model, criterion, optimizer, num_epochs=5):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # print(labels)
                    # print(inputs)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.detach() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.float() / len(image_datasets[phase])

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss.item(),
                                                            epoch_acc.item()))
        
        return model
    

    
    model_trained = train_model(model, criterion, optimizer, num_epochs=5)
