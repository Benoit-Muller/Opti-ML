import torch
import copy
from torchvision import models
import torch.nn as nn
from tqdm import tqdm,trange

def set_parameter_requires_grad(model, req_grad):
    if not req_grad:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(input_size=32,input_channel=3, output_size=10, req_grad=True, use_pretrained=True):

    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, req_grad)
    num_ftrs = model_ft.fc.in_features #look at the number of feature at the entry of last layer of resnet
    model_ft.fc = nn.Linear(num_ftrs, output_size) #modify the last layer

    #Replace the first layer ?
    #conv1_out_channel = model_ft.conv1.out_channels
    #model_ft.conv1 = nn.Conv2d(input_channel, conv1_out_channel, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model_ft, input_size

def train_epoch(model, trainloader, criterion, optimizer,device='cpu'):
  #train for one epoch
  model.train()
  running_loss = 0.0
  for i, data in enumerate(trainloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)
      inputs = inputs.float()

      # zero the parameter gradients
      optimizer.zero_grad()

      with torch.set_grad_enabled(True):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(create_graph=True) #needed for Hessian backprog
        optimizer.step()

      # print statistics
      running_loss += loss.item() * inputs.size(0)
  epoch_loss = running_loss / len(trainloader.dataset) #average
  print('Train Loss: {:.4f}'.format(epoch_loss))
  return epoch_loss

def test_loss(model, testloader, criterion, optimizer, epoch, num_epoch, best_loss, best_model_wts,savepath,device):
    #compute the loss on a test set
    model.eval()
    running_loss = 0.0
    for i, data in enumerate(testloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(testloader.dataset)
    print('Test Loss: {:.4f}'.format(epoch_loss))
    if(epoch_loss < best_loss and epoch > num_epoch/2): #save the best model
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, savepath)
    return epoch_loss, best_loss, best_model_wts

def train_and_test(model, trainloader,testloader, criterion, optimizer,num_epoch,savepath,device='cpu'):
    #train and test for each epoch
    #trainloader is a Dataloader
    train_loss=[]
    list_testloss=[]
    lowest_loss = 100000
    model_weights = None
    for epoch in trange(num_epoch):
        if(epoch%1==0):
            print('Epoch {}/{}'.format(epoch, num_epoch-1))
            print('-' * 10)
        trainloss=train_epoch(model, trainloader, criterion, optimizer, device)
        testloss,lowest_loss, model_weights = test_loss(model, testloader, criterion, optimizer, epoch, num_epoch, lowest_loss, model_weights,savepath,device)
        train_loss.append(trainloss)
        list_testloss.append(testloss)
    return train_loss, list_testloss, lowest_loss, model_weights

def test_model(model,testloader,model_path,device):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    num_cor = 0
    for i, data in enumerate(testloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()

        prediction = model(inputs)
        prediction = torch.argmax(prediction)
        #print("gt:", labels, "pred:", prediction)
        if(labels == prediction):
            num_cor += 1
    acc = num_cor / len(testloader.dataset)
    return acc