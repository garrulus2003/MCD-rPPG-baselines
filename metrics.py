import numpy as np

def mse(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def mae(y_true, y_pred):
    return (np.abs(y_true - y_pred)).mean()

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()
    
def F1(y_true, y_pred, one_as_pos=True):    
    if not one_as_pos:
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    precision = ((y_true * y_pred).sum() / y_pred.sum())
    recall = ((y_true * y_pred).sum() / y_true.sum())
    return 2 * recall * precision / (recall + precision)