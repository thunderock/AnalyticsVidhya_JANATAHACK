
import numpy as np
import pandas as pd
import pickle as pkl
import sys



pred = pkl.load(open('val_preds.pkl', 'rb'))

y = pkl.load(open("val_original.pkl", 'rb'))
y = y[0:0 + pred.shape[0]]

thresholds = [0] * 25


def get_current_loss(p_col, true_col, th):
    p_col = sum(p_col > th)
    true_col = sum(true_col > th)
    print(p_col, true_col, th)
    return abs(p_col - true_col)

for col in range(25):
    current_loss = sys.maxsize

    for threshold in np.linspace(0, 1, 20, endpoint=False):
        pred_col = np.take(pred, 0, axis=1)
        y_col = np.take(y, 0, axis=1)
        if get_current_loss(pred_col, y_col, threshold) < current_loss:
            thresholds[col] = threshold
