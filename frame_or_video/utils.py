import torch
import numpy as np
import pandas as pd
from metrics import mse, mae, accuracy, F1
from utils import rescale, TARGETS
from tqdm.auto import tqdm 

def train_epoch(model, dl, optimizer, criterion, is_regression=True, 
                device='cpu', scale=1, all_targets=False):
    model.to(device)
    model.train()

    loss_history = []

    y_true = []
    y_pred = []
    
    for batch in dl:
        x = batch[0].to(device)
        y = batch[1].to(device)
        
        y_true += list(batch[1])
        outputs = model(x)

        if is_regression:
            y_pred += list(outputs.cpu())
            
            if all_targets:
                loss = criterion(outputs, y)
            else:
                loss = criterion(outputs.reshape(-1), y)
                
        else:
            y_pred += list(outputs.argmax(axis=1).cpu())
            loss = criterion(outputs, y)
            
        loss_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    if all_targets:
        y_true = torch.stack(y_true).detach().cpu().numpy()
        y_pred = torch.stack(y_pred).detach().cpu().numpy()
        
        return rescale(y_true), rescale(y_pred), loss_history
        
    y_true = torch.tensor(y_true).detach().cpu().numpy()
    y_pred = torch.tensor(y_pred).detach().cpu().numpy()

    return y_true * scale, y_pred * scale, loss_history


def eval_net(model, dl, is_regression=True, 
             device='cpu', scale=1, all_targets=False):
    
    model.to(device)
    model.eval()
    y_pred = []
    y_true = []
    videos = []

    metrics = dict()
    
    with torch.no_grad():
        for batch in tqdm(dl):
            x = batch[0].to(device)
            y_true += list(batch[1])
            videos += list(batch[2])
            outputs = model(x)

            if is_regression:
                y_pred += list(outputs.cpu())
            else:
                y_pred += list(outputs.argmax(axis=1).cpu())
                
        if all_targets:
            y_true = torch.stack(y_true).detach().cpu().numpy()
            y_pred = torch.stack(y_pred).detach().cpu().numpy()
        
            y_true = rescale(y_true)
            y_pred = rescale(y_pred)

            predictions = {'video' : videos}
            for j, target in enumerate(TARGETS):
                metrics[target + '_mse_train'] = str(mse(y_true[:, j], y_pred[:, j]))
                metrics[target + '_mae_train'] = str(mae(y_true[:, j], y_pred[:, j]))
                predictions[target + '_true'] = y_true[:, j]
                predictions[target + '_pred'] = y_pred[:, j]

            
        else:
            y_true = torch.tensor(y_true).detach().cpu().numpy()
            y_pred = torch.tensor(y_pred).detach().cpu().numpy()
            y_true = y_true * scale
            y_pred = y_pred * scale

            if is_regression:
                metrics['mse_train'] = str(mse(y_true, y_pred))
                metrics['mae_train'] = str(mae(y_true, y_pred))
            else:
                metrics['acc_val'] = str(accuracy(y_true, y_pred))
                metrics['f1_pos_val'] = str(F1(y_true, y_pred))
                metrics['f1_neg_val'] = str(F1(y_true, y_pred, one_as_pos=False))

            predictions = {
                'video' : videos,
                'true': y_true,
                'predicted': y_pred
            }
        
        return pd.DataFrame(predictions), metrics





def train_net(model, dl, optimizer, criterion, num_epochs, 
              is_regression=True, device='cpu',
              scale=1, all_targets=False):
    if all_targets:
        results = {}
        results['loss_history'] = []

        for target in TARGETS:
            results[target + '_mse_train'] = []
            results[target + '_mae_train'] = []

        for i in tqdm(range(num_epochs)):
            y_true, y_pred, loss_history = train_epoch(
                model, dl, 
                optimizer, criterion, 
                is_regression=is_regression, 
                device=device,
                scale=scale,
                all_targets=all_targets
            )
            
            results['loss_history'].append(
                np.array(loss_history).mean()
            )

            for j, target in enumerate(TARGETS):
                results[target + '_mse_train'].append(
                    mse(y_true[:, j], y_pred[:, j])
                )
                results[target + '_mae_train'].append(
                    mae(y_true[:, j], y_pred[:, j])
                )
                
        return results
            
    if is_regression:
        results = {
            'loss_history' : [],
            'mse_train' : [],
            'mae_train' : []
        }
        
        for i in tqdm(range(num_epochs)):
            y_true, y_pred, loss_history = train_epoch(
                model, dl, 
                optimizer, criterion, 
                is_regression=is_regression, 
                device=device,
                scale=scale)

            results['loss_history'].append(np.array(loss_history).mean())
            results['mse_train'].append(mse(y_true, y_pred))
            results['mae_train'].append(mae(y_true, y_pred))

    else:

        results = {
            'loss_history' : [],
            'acc_train' : [],
            'f1_pos_train' : [],
            'f1_neg_train' : [],
        }

        for i in tqdm(range(num_epochs)):
            y_true, y_pred, loss_history = train_epoch(
                model, dl, 
                optimizer, criterion, 
                is_regression=is_regression, 
                device=device,
                scale=scale
            )
            
            results['loss_history'].append(np.array(loss_history).mean())
            results['acc_train'].append(accuracy(y_true, y_pred))
            results['f1_pos_train'].append(F1(y_true, y_pred))
            results['f1_neg_train'].append(F1(y_true, y_pred, one_as_pos=False))

    return results
