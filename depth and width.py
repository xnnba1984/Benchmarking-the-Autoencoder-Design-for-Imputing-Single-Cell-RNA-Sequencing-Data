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

# hidden layer width
hidden = [32, 64, 128, 256]

# three types of masking data
mask_type = 'masked dataset median/'
save_type = 'imputed dataset median/'

# train autoencoder and impute every dataset
for d in datasets:
    for h in hidden:
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('processing dataset: ' + d + ' with hidden width ' + str(h))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        train(dataset=d, H=h, repeat=10, seed=2020, epoch=10000, learning_rate=1e-3, layer_total=15, 
              batchsize=64)

# the autoencoder training function
# the training stops when validation MSE does not decrease over 20 epochs or the total number of epochs surpasses 10000
# the final autoencoder will impute the whole data
def train(dataset, H, seed=2020, layer_total = 15, learning_rate = 1e-4, epoch = 1000, batchsize = 64, 
                  no_improve_limit = 20, repeat = 10):

    # read dataset
    train = mmread(mask_type + dataset + '_mask_train.mtx')
    test = mmread(mask_type + dataset + '_mask_test.mtx')

    # format
    train = train.todense()
    test = test.todense()
    train = np.array(train)
    test = np.array(test)
    train = train.T
    test = test.T
    train_tensor = torch.tensor(train, dtype=torch.float)
    test_tensor = torch.tensor(test, dtype=torch.float)

    # model
    torch.manual_seed(seed)
    np.random.seed(seed)

    # global model parameter
    D_in = train_tensor.shape[1]
    D_out = D_in

    # automatically build models with different layers
    models = []
    for layer in np.arange(layer_total):
        layers = []
        layers.append(nn.Linear(D_in, H))
        layers.append(nn.ReLU())
        if layer >= 1:
            for i in range(layer):
                layers.append(nn.Linear(H, H))
                layers.append(nn.ReLU()) 
        layers.append(nn.Linear(H, D_out)) 
        model = nn.Sequential(*layers)
        models.append(model)

    # glaboal control parameter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.MSELoss()
    row = np.arange(train_tensor.shape[0])

    # final data to predict
    data = np.concatenate((train, test), axis=0)
    data_tensor = torch.tensor(data, dtype=torch.float)
    data_tensor = data_tensor.to(device)

    # loop over each model
    for k, model in enumerate(models):
        print('**********************************')
        print("model", k + 1)
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
                # no improvement count
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

            # use trained model to impute
            best_model.eval()
            pred = best_model(data_tensor)
            pred = pred.cpu().detach().numpy()
            save_file = save_type + dataset + '_h' + str(k+1) + '_' + str(H) + '_' + str(i+1) + '.npy'
            print('save to:', save_file)
            np.save(save_file, pred)
            
