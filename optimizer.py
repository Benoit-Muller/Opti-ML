import torch
import copy
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm,trange
import sys
sys.path.insert(0,'/content/drive/MyDrive/Opti-ML') #to access other files
sys.path.insert(0,"adahessian/image_classification/")
from optim_adahessian import Adahessian

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

def test_loss(model, testloader, criterion, optimizer, epoch, num_epoch,device):
    #model, testloader, criterion, optimizer, epoch, num_epoch, best_loss, best_model_wts,savepath,fold,device
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
    epoch_loss = 4 * running_loss / len(testloader.dataset)
    print('Validation Loss: {:.4f}'.format(epoch_loss))
    #if(epoch_loss < best_loss and epoch > num_epoch/2): #save the best model
        #best_loss = epoch_loss
        #best_model_wts = copy.deepcopy(model.state_dict())
        #torch.save(best_model_wts, f'{savepath}_fold{fold}.pt')
    return epoch_loss

def train_get_acc(model, trainloader, device):
    model.eval()
    num_cor = 0
    num_all = 0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()

        prediction = model(inputs)
        #prediction = prediction.tolist()
        #print("pred:", prediction[0])
        labels =  labels.tolist()
        #print("length of labels:", len(labels))
        for i in range(len(labels)):
            num_all += 1
            #print(labels[i])
            pred = int(torch.argmax(prediction[i]))
            #print(pred)
            if(labels[i] == pred):
                num_cor += 1
    acc = num_cor / num_all
    return acc

def train_and_test(model,traindataset,batch_size,kfold,criterion,optimizer_name,scheduler,num_epoch,savepath,use_pretrained,lr,betas,momentum,weight_decay,hessian_power,device='cpu'):
    #train and test for each epoch
    #trainloader is a Dataloader
    train_loss=[]
    train_acc_list=[]
    val_acc_list=[]
    val_loss=[]
    #lowest_loss = 100000
    #model_weights = None
    best_val_acc = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(traindataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, sampler=train_subsampler)
        valloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, sampler=val_subsampler)
        model, input_size = initialize_model(use_pretrained=use_pretrained)
        model = model.to(device)
        if optimizer_name == 'adahessian':
          optimizer = Adahessian(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay,hessian_power=hessian_power) #set the parameter
        elif optimizer_name == 'SGD':
          optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "adam":
          optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        k_train_loss=[]
        k_train_acc_list=[]
        k_val_acc_list=[]
        k_val_loss=[]
        k_best_val_acc=0
        for epoch in trange(num_epoch):
            #if(epoch%5==1):
            print('Epoch {}/{}'.format(epoch, num_epoch-1))
            print('-' * 10)
            
            trainloss=train_epoch(model, trainloader, criterion, optimizer, device)
            train_acc=train_get_acc(model, trainloader, device)
            
            valloss=test_loss(model, valloader, criterion, optimizer, epoch, num_epoch,device)
            val_acc=train_get_acc(model, valloader, device)
            if(val_acc > k_best_val_acc and epoch > num_epoch/2):
                k_best_val_acc = val_acc
                torch.save(model.state_dict(), f'{savepath}_fold{fold}.pt')
            
            print('Train accuracy: {:.4f}'.format(train_acc))
            print('Validation accuracy: {:.4f}'.format(val_acc))
                
            k_train_loss.append(trainloss)
            k_train_acc_list.append(train_acc)
            k_val_acc_list.append(val_acc)
            k_val_loss.append(valloss)

            if scheduler is not None:
                scheduler.step()
        train_loss.append(k_train_loss)
        train_acc_list.append(k_train_acc_list)
        val_acc_list.append(k_val_acc_list)
        val_loss.append(k_val_loss)
        best_val_acc.append(k_best_val_acc)
    best_k = best_val_acc.index(max(best_val_acc))
    best_train_acc = train_acc_list[best_k]
    best_train_loss = train_loss[best_k]
    best_val_loss = val_loss[best_k]
    best_val_acc = val_acc_list[best_k]

    return best_k,best_train_loss, best_val_loss, best_train_acc, best_val_acc

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