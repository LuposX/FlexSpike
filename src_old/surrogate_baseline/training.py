import numpy as np
import random
# import matplotlib.pyplot as plt
import torch
import calendar
import time
import config
import math
import os

def train_nn(nn, train_loader, valid_loader, lossfunction, optimizer, UUID='default'):
    
    training_ID = int(calendar.timegm(time.gmtime()))
    if not UUID == 'default':
        UUID = f'{hash(UUID)}'
    print(f'The ID for this training is {UUID}_{training_ID}.')
    
    train_loss = []
    valid_loss = []
    best_valid_loss = math.inf
    patience = 0

    for epoch in range(10**10):
        time_start = time.time()
        
        total_loss = 0.0
        total_samples = 0
        for x_train, y_train in train_loader:
            prediction_train = nn(x_train)
            L_train = lossfunction(prediction_train, y_train)
            
            total_loss += L_train.item() * x_train.size(0)
            total_samples += x_train.size(0)
            
            optimizer.zero_grad()
            L_train.backward()
            optimizer.step()

        weighted_mean_loss = total_loss / total_samples
        train_loss.append(weighted_mean_loss)
            
        # Validation part    
        total_valid_loss = 0.0
        total_valid_samples = 0
        nn.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                prediction_valid = nn(x_valid)
                L_valid = lossfunction(prediction_valid, y_valid)
                total_valid_loss += L_valid.item() * x_valid.size(0)
                total_valid_samples += x_valid.size(0)
        weighted_mean_valid_loss = total_valid_loss / total_valid_samples
        valid_loss.append(weighted_mean_valid_loss)


        if L_valid.item() < best_valid_loss:
            best_valid_loss = L_valid.item()
            torch.save(nn, f'./temp/NN_{UUID}_{training_ID}')
            patience = 0
        else:
            patience += 1

        if patience > 100:
            print('Early stop.')
            break

        # if not epoch % 500:
        time_end = time.time()
        print(f'| Epoch: {epoch:-8d} | Train loss: {L_train.item():.5f} | Valid loss: {L_valid.item():.5f} | Patience: {patience} | Epoch Time: {time_end-time_start:.2f} |')
    
    # remove temp files
    resulted_nn = torch.load(f'./temp/NN_{UUID}_{training_ID}', weights_only=False)
    os.remove(f'./temp/NN_{UUID}_{training_ID}')
    
    print('Finished.')
    return resulted_nn, train_loss, valid_loss