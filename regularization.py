import torch
import torch.nn as nn
import copy
import numpy as np
from scipy.io import mmread
from scipy.io import mmwrite
from sys import getsizeof

# user specificed dataset name vector used to train autoencoder and impute
datasets = ['jurkat', 'monocyte','mbrain','pbmc','lymphoma','293t','bmmc','human_mix','mouse_spleen',
            'mouse_cortex', 'mouse_skin', 'cbmc']

# weight decay parameter lambd
wds = np.array([1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4])

# autoencoder hyperparameters
D_in = 2000
D_out = 2000
H = 32
layer_total = 10
act = nn.Sigmoid()

# droput ratess
dp = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
models = []
seed = 2020
repeat = 5

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# define the model without regularization as baseline
layers = []
layers.append(nn.Linear(D_in, H))
layers.append(act)
for i in range(layer_total-1):
    layers.append(nn.Linear(H, H))
    layers.append(act) 
layers.append(nn.Linear(H, D_out)) 
model = nn.Sequential(*layers)
models.append(model)


# pre-build models with different dropout rates
for p in dp: 
    layers = []
    layers.append(nn.Linear(D_in, H))
    layers.append(act)
    layers.append(nn.Dropout(p=p))
    for i in range(layer_total-1):
        layers.append(nn.Linear(H, H))
        layers.append(act) 
        layers.append(nn.Dropout(p=p))
    layers.append(nn.Linear(H, D_out)) 
    model = nn.Sequential(*layers)
    models.append(model)

# loop over all datasets    
for dataset in datasets:
    # read dataset
    train_data = mmread('masked dataset/' + dataset + '_mask_train.mtx')
    test_data = mmread('masked dataset/' + dataset + '_mask_test.mtx')

    # train autoencoder with different weight decays and then impute
    for wd in wds:
        print('********************************************************************')
        print('processing dataset: ' + dataset + ' with weight_decay ' + str(wd))
        print('********************************************************************')
        # repeat
        for i in range(repeat):
            print('=================================')
            print('repeat: ', i + 1)
            pred = train(train=train_data, test=test_data, model=models[0], wd=wd)                             
            save_file = 'imputed dataset/' + dataset + '_wd_' + str(wd) + '_' + str(i+1) + '.npy'
            print('save to:', save_file)
            np.save(save_file, pred)

    # train autoencoder with different dropout rates and then impute
    for model, p in zip(models, dp):
        print('********************************************************************')
        print('processing dataset: ' + dataset + ' with dropout ' + str(p))
        print('********************************************************************')
        # repeat
        for i in range(repeat):
            print('=================================')
            print('repeat: ', i + 1)
            pred = train(train=train_data, test=test_data, model=model, wd=0)                            
            save_file = 'imputed dataset/' + dataset + '_dp_' + str(p) + '_' + str(i+1) + '.npy'
            print('save to:', save_file)
            np.save(save_file, pred)
         
# the autoencoder training function
# the training stops when validation MSE does not decrease over 20 epochs or the total number of epochs surpasses 10000
# the final autoencoder will impute the whole data
def train(train, test, model, wd = 0, learning_rate = 1e-4, epoch = 1000, batchsize = 64, no_improve_limit = 20):
    # format
    train = train.todense()
    test = test.todense()
    train = np.array(train)
    test = np.array(test)
    train = train.T
    test = test.T
    train_tensor = torch.tensor(train, dtype=torch.float)
    test_tensor = torch.tensor(test, dtype=torch.float)

    # glaboal control parameter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = torch.nn.MSELoss()
    row = np.arange(train_tensor.shape[0])

    # final data to predict
    data = np.concatenate((train, test), axis=0)
    data_tensor = torch.tensor(data, dtype=torch.float)
    data_tensor = data_tensor.to(device)    
    
    model_repeat = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_repeat.parameters(), lr=learning_rate, weight_decay=wd)
    best_test_loss = 1e4
    count = 1
    best_epoch = 1
    best_model = model_repeat
    # loop over epoch
    for e in range(epoch):
        #print('epoch:', e + 1)
        np.random.shuffle(row)
        batch_index = [row[i:i + batchsize] for i in range(0, len(row), batchsize)] 
        running_loss = 0
        # train loop over batches
        model_repeat.train()
        for b in batch_index:
            batch_tensor = train_tensor[b,]
            mask = batch_tensor != 0
            mask = mask.to(device)
            mask = mask.float()
            batch_tensor = batch_tensor.to(device)
            y_pred = model_repeat(batch_tensor)
            y_pred = y_pred * mask
            loss = loss_fn(batch_tensor, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item() * batch_tensor.shape[0]
        train_epoch_loss = running_loss / train_tensor.shape[0]
        #print('train_epoch_loss: ', train_epoch_loss)

        # test
        model_repeat.eval()
        mask = test_tensor != 0
        mask = mask.to(device)
        mask = mask.float()
        test_tensor = test_tensor.to(device)
        y_pred = model_repeat(test_tensor)
        y_pred = y_pred * mask
        loss = loss_fn(test_tensor, y_pred).item()
        # save best loss
        if loss < best_test_loss:
            best_test_loss = loss
            best_epoch = e + 1
            count = 1
            best_model = model_repeat
        else:
            count = count + 1     
        #print('test_epoch_loss: ', loss)
        if count > no_improve_limit:
            break

    print('best_test_loss: ', best_test_loss)
    print('best_epoch: ', best_epoch)

    # use trained model to predict
    best_model.eval()
    pred = best_model(data_tensor)
    pred = pred.cpu().detach().numpy()
    return pred