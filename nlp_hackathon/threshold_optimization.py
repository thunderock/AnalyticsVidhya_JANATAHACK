import numpy as np
import pandas as pd
import pickle as pkl
import sys
import pandas as pd


pred = '/tmp/val_preds1.pkl'
y = '/tmp/val_original1.pkl'
final_pred = '/tmp/pred_proba1.pkl'


pred = pkl.load(open(pred, 'rb'))

y = pkl.load(open(y, 'rb'))
y = y[0:0 + pred.shape[0]]

thresholds = [0] * 25
counts = [0] * 25
y_trues = [0] * 25

def get_current_loss(p_col, true_col, th):
    p_col = sum(p_col > th)
    true_col = sum(true_col > th)
    return abs(p_col - true_col)

for col in range(25):
    current_loss = sys.maxsize
    temp_loss = sys.maxsize
    for threshold in np.linspace(0, 1, 20, endpoint=False):
        pred_col = np.take(pred, col, axis=1)
        y_col = np.take(y, col, axis=1)
        temp_loss = get_current_loss(pred_col, y_col, threshold)
        if  temp_loss < current_loss:
            thresholds[col] = threshold
            current_loss = temp_loss
            counts[col] = sum(pred_col > threshold)
            y_trues[col] = sum(y_col > threshold)
            # print(col, current_loss, thresholds[col], counts[col], y_trues[col])

print(thresholds)
print(counts)
print(y_trues)

df = pd.read_csv('Train.csv')

y = df.iloc[:, 6: 6 + 25]
LABELS = y.columns


tdf = pd.read_csv("Test.csv")
i = pkl.load(open(final_pred, "rb"))

for col in range(len(LABELS)):
    tdf[LABELS[col]] = [1 if x[col] > thresholds[col] else 0 for x in i]
    

tdf.drop(columns=['ABSTRACT', 'Computer Science', 'Mathematics', 'Physics', 'Statistics']).to_csv("/tmp/final.csv", index=False)

