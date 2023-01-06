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

# activation functions to compare
af = [nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.LeakyReLU(negative_slope=0.2), nn.ELU(), nn.SELU()]

# train autoencoder and impute across every dataset
for d in datasets:
    for a in af:
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('processing dataset: ' + d + ' with activation function ' + str(a))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        train(dataset=d, H=32, act=a, repeat=20, seed=2020, layer_total=10)
        
# train autoencoder and impute every dataset
def train(dataset, H, act, seed=2020, layer_total = 15, learning_rate = 1e-4, epoch = 1000, batchsize = 64, 
                  no_improve_limit = 20, repeat = 10):

    # read dataset
    train = mmread('masked dataset/' + dataset + '_mask_train.mtx')
    test = mmread('masked dataset/' + dataset + '_mask_test.mtx')

    # format
    train = train.todense()
    test = test.todense()
    train = np.array(train)
    test = np.array(test)
    train = train.T
    test = test.T
    train_tensor = torch.tensor(train, dtype=torch.float)
    test_tensor = torch.tensor(test, dtype=torch.float)

    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # global model parameter
    D_in = train_tensor.shape[1]
    D_out = D_in

    # automatically build models
    layers = []
    layers.append(nn.Linear(D_in, H))
    layers.append(act)
    for i in range(layer_total-1):
        layers.append(nn.Linear(H, H))
        layers.append(act) 
    layers.append(nn.Linear(H, D_out)) 
    model = nn.Sequential(*layers)

    # glaboal control parameter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.MSELoss()
    row = np.arange(train_tensor.shape[0])

    # final data to predict
    data = np.concatenate((train, test), axis=0)
    data_tensor = torch.tensor(data, dtype=torch.float)
    data_tensor = data_tensor.to(device)

    print('**********************************')
    print("model", str(act))
    print('**********************************')
    model = model.to(device)
    # repeat
    for i in range(repeat):
        model_repeat = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model_repeat.parameters(), lr=learning_rate)
        best_test_loss = 1e4
        count = 1
        best_epoch = 1
        best_model = model_repeat
        print('=================================')
        print('repeat: ', i + 1)
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
        save_file = 'imputed dataset/' + dataset + '_h' + str(layer_total) + '_' + str(act) + \
                    '_' + str(H) + '_' + str(i+1) + '.npy'
        print('save to:', save_file)
        np.save(save_file, pred)