
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from utilities.util import ensure_dir

import numpy as np


dataset = "yacht"

TEST_DATA_RATIO = 0.1

assert(TEST_DATA_RATIO == 0.1)


filename = f'./data/{dataset}/{dataset}.csv'
save_path = f'./gap_data/{dataset}_gap'

df = pd.read_csv(filename, header=None)

last_column_id = len(df.columns) - 1
X = df.drop(columns=last_column_id, axis=1)
y = df.loc[:, last_column_id]

X = X.to_numpy()
y = y.to_numpy()

NR_FOLDS = X.shape[1]

print("NR_FOLDS = ", NR_FOLDS)

for foldId in range(1, 1 + NR_FOLDS):

    # GAP split as described in "‘In-Between’ Uncertainty in Bayesian Neural Networks", 2019

    ids = np.argsort(X[:, foldId - 1])

    n = ids.shape[0]
    train_half_size = int(n * (1.0 - TEST_DATA_RATIO) / 2.0)

    train_lower_ids = ids[0:train_half_size]
    train_upper_ids = ids[(n - train_half_size):n]

    test_ids = ids[train_half_size:(n - train_half_size)]
    train_ids = np.concatenate((train_lower_ids, train_upper_ids))
    
    train_X = X[train_ids]
    train_y = y[train_ids]
    test_X = X[test_ids]
    test_y = y[test_ids]
    
    print(train_lower_ids.shape)
    print(train_upper_ids.shape)
    print(test_ids.shape)
    print("train_ids = ", train_ids.shape)

    print("y = ", y.shape)
    print("train_y = ", train_y.shape)
    print("test_y = ", test_y.shape)

    print("X = ", X.shape)
    print("train_X = ", train_X.shape)
    print("test_X = ", test_X.shape)
    
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=foldId, shuffle=True)
         
    PATH = os.path.join(save_path, str(foldId))

    ensure_dir(PATH)    

    print("write train")
    # csvファイルとして保存
    np.savetxt(os.path.join(PATH, 'train_X.csv'), train_X, delimiter=",")
    np.savetxt(os.path.join(PATH, 'train_y.csv'), train_y, delimiter=",")

    print("write val")
    np.savetxt(os.path.join(PATH, 'val_X.csv'), val_X, delimiter=",")
    np.savetxt(os.path.join(PATH, 'val_y.csv'), val_y, delimiter=",")

    print("write test")
    np.savetxt(os.path.join(PATH, 'test_X.csv'), test_X, delimiter=",")
    np.savetxt(os.path.join(PATH, 'test_y.csv'), test_y, delimiter=",")


print("FINISHED SUCCESSFULLY")
