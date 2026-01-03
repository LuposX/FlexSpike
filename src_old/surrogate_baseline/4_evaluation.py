


import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import config
import os

models = ['gpt-nano', 'gpt-micro', 'gpt-mini', 'gopher-44m', 'gpt2']

data = torch.load(f'./data/dataset.ds')
X_valid = data['X_valid']
Y_valid = data['Y_valid']
X_test  = data['X_test']
Y_test  = data['Y_test']
valid_data = TensorDataset(X_valid, Y_valid)
test_data  = TensorDataset(X_test, Y_test)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

results = torch.zeros([5, 2])

seed = int(sys.argv[1])
model_idx = int(sys.argv[2])

for l, lr in enumerate([-2,-3,-4,-5,-6]):

    modelname = f'predictor_{models[model_idx]}_lr_{lr}_seed_{seed}'
    if os.path.isfile(f'./NNs/{modelname}'):

        model = torch.load(f'./NNs/{modelname}', map_location=torch.device('cpu'), weights_only=False)

        config.SetSeed(seed)

        loss_fn = torch.nn.MSELoss(reduction='mean')

        total_loss = 0.0
        total_samples = 0
        for x_valid, y_valid in valid_loader:
            prediction_valid = model(x_valid)
            L_valid = loss_fn(prediction_valid, y_valid)
            total_loss += L_valid.item() * x_valid.size(0)
            total_samples += x_valid.size(0)
        weighted_mean_loss = total_loss / total_samples
        results[l, 0] = weighted_mean_loss

        total_loss = 0.0
        total_samples = 0
        for x_test, y_test in test_loader:
            prediction_test = model(x_test)
            L_test = loss_fn(prediction_test, y_test)
            total_loss += L_test.item() * x_test.size(0)
            total_samples += x_test.size(0)
        weighted_mean_loss = total_loss / total_samples
        results[l, 1] = weighted_mean_loss

        print(results[l, 0], results[l, 1])
    else:
        results[l, :] = float('nan')
                
torch.save(results, f'./data/SG_results_{seed}_{model_idx}.matrix')
