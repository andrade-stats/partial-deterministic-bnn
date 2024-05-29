
import os

import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from utilities.util import ensure_dir


parser = ArgumentParser()
parser.add_argument('--test_size', type=lambda x: int(x) if x.isdigit() else float(x),
                    help='an integer or a float')
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

test_size = args.test_size
dataset = args.dataset

assert(test_size == 0.1)

if (type(test_size) is float) and (0. < test_size) and (test_size > 1.):
    raise ValueError("float型の場合、0.0〜1.0未満の値を入力してください。")


filename = f'./data/{dataset}/{dataset}.csv'
save_path = f'./data/{dataset}'

df = pd.read_csv(filename, header=None)


last_column_id = len(df.columns) - 1
X = df.drop(columns=last_column_id, axis=1)
y = df.loc[:, last_column_id]


if (dataset == "protein") or (dataset == "miniboo"):
    NR_FOLDS = 5
else:
    NR_FOLDS = 10


for seed in range(1, 1 + NR_FOLDS):
    train_X, test_X, train_y, test_y = train_test_split(X, y ,test_size=test_size,random_state=seed, shuffle=True)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=seed, shuffle=True)
    
    PATH = os.path.join(save_path, str(seed))

    ensure_dir(PATH)    

    print("write train")
    # csvファイルとして保存
    train_X.to_csv(os.path.join(PATH, 'train_X.csv'), header=None, index=None)
    train_y.to_csv(os.path.join(PATH, 'train_y.csv'), header=None, index=None)

    print("write val")

    val_X.to_csv(os.path.join(PATH, 'val_X.csv'), header=None, index=None)
    val_y.to_csv(os.path.join(PATH, 'val_y.csv'), header=None, index=None)

    print("write test")

    test_X.to_csv(os.path.join(PATH, 'test_X.csv'), header=None, index=None)
    test_y.to_csv(os.path.join(PATH, 'test_y.csv'), header=None, index=None)    

