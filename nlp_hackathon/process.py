
import numpy as np
import pandas as pd
import pickle as pkl


filename = "i1.pkl"
THRESHOLD = .15

df = pd.read_csv('Train.csv')

X = df.iloc[:, 1: 6]
y = df.iloc[:, 6: 6 + 25]

LABELS = y.columns



y = y.values

tdf = pd.read_csv("Test.csv")
i = pkl.load(open(filename, "rb"))


cnt = {}
for x in y:
    ones = 0
    for xx in x:
        if xx == 1: ones += 1
    if ones in cnt: cnt[ones] += 1
    else: cnt[ones] = 1
for x in cnt: 
    print(x, cnt[x] / sum(cnt.values()))


print("------------------------")

cnt2 = {}
for x in i:
    ones = 0
    for xx in x:
        if xx > THRESHOLD: ones += 1
    if ones in cnt2: cnt2[ones] += 1
    else: cnt2[ones] = 1
for x in cnt2: 
    print(x, cnt2[x] / sum(cnt2.values()))



for col in range(len(LABELS)):
    tdf[LABELS[col]] = [1 if x[col] > THRESHOLD else 0 for x in i]
    


tdf.drop(columns=['ABSTRACT', 'Computer Science', 'Mathematics', 'Physics', 'Statistics']).to_csv("final.csv", index=False)

