import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ppg_models.hr_utils import calculate_metric_per_video


def train_epoch(model, train_loader, config, optimizer, scheduler, criterion, device):
    model.train()
    model.to(device)
    train_loss = []
    hrs_pred = []
    hrs_true = []
    
    for batch in train_loader:
        frames, bvp, hr = batch
        frames = frames.to(torch.float)
        bvp = bvp.to(torch.float)
        frames = frames.to(device)
        bvp = bvp.to(device)
        hrs_true += list(hr)
        frames = frames.permute(0, 1, 4, 2, 3)
        frames = frames.view(-1, 6, config['H'], config['W'])
        bvp = bvp.view(-1, 1)
        optimizer.zero_grad()
        pred_ppg = model(frames)
        loss = criterion(pred_ppg, bvp)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())
        pred_ppg = pred_ppg.reshape(-1, len(pred_ppg)//config['BATCH_SIZE'])
        for ppg in pred_ppg:
            hrs_pred.append(
                calculate_metric_per_video(
                    ppg.detach().cpu().numpy(), 
                    diff_flag=True
                )
            )

    hr_mae = np.abs(np.array(hrs_true) - np.array(hrs_pred)).mean()

    print(hrs_pred)
    print(hrs_true)


    return np.array(train_loss).mean(), hr_mae


def eval_net(model, val_loader, config, device):
    criterion = torch.nn.MSELoss()
    model.eval()
    model.to(device)
    val_loss = []
    hrs_pred = []
    hrs_true = []
    videos = []

    
    for batch in tqdm(val_loader):
        frames, bvp, hr, video_path = batch
        frames = frames.to(torch.float)
        bvp = bvp.to(torch.float)
        frames = frames.to(device)
        bvp = bvp.to(device)
        hrs_true += list(hr)
        videos += list(video_path)
        frames = frames.permute(0, 1, 4, 2, 3)
        frames = frames.view(-1, 6, config['H'], config['W'])
        bvp = bvp.view(-1, 1)
        pred_ppg = model(frames)
        loss = criterion(pred_ppg, bvp)
        val_loss.append(loss.item())
        pred_ppg = pred_ppg.reshape(1, len(pred_ppg))
        for ppg in pred_ppg:
            hrs_pred.append(
                calculate_metric_per_video(
                    ppg.detach().cpu().numpy(), 
                    diff_flag=True
                )
            )

    hr_mae = np.abs(np.array(hrs_true) - np.array(hrs_pred)).mean()

    predictions = {
        'video': videos,
        'hr_true': np.array(hrs_true),
        'hr_pred': hrs_pred
    }

    metrics = {
        'loss': str(np.array(val_loss).mean()),
        'hr_mae' : str(hr_mae)
    }
    return pd.DataFrame(predictions), metrics
    

def train_net(model, train_loader, config, device):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['LR'], 
        weight_decay=0
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['LR'], 
        epochs=config['EPOCHS'], 
        steps_per_epoch=len(train_loader)
    )

    TRAIN_LOSS = []
    VAL_LOSS = []
    HR_MAE = []

    for epoch in tqdm(range(config['EPOCHS'])):
        train_loss, hr_mae = train_epoch(
            model, train_loader, config, 
            optimizer, scheduler, criterion, device
        )
        TRAIN_LOSS.append(train_loss)
        HR_MAE.append(hr_mae)

    results = {
        'train_loss': TRAIN_LOSS,
        'hr_mae': HR_MAE
    }
    
    return results