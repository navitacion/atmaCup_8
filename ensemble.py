import os, glob
import pandas as pd



def ensemble_weight(csv_list, weights):
    target_col = 'Global_Sales'
    filename = ''
    for i, (path, w) in enumerate(zip(csv_list, weights)):
        temp = pd.read_csv(path)

        if i == 0:
            res = temp.copy()
            res[target_col] = w * res[target_col]
        else:
            res[target_col] += w * temp[target_col]

    res.to_csv(f'ensemble_mean.csv', index=False)


def ensemble_mean(csv_list):
    target_col = 'Global_Sales'
    filename = ''
    for i, path in enumerate(csv_list):
        temp = pd.read_csv(path)

        if i == 0:
            res = temp.copy()
        else:
            res[target_col] += temp[target_col]

    res[target_col] = res[target_col]/ len(csv_list)

    res.to_csv(f'ensemble_mean.csv', index=False)


if __name__ == '__main__':
    csv_list = glob.glob('./ensemble/*.csv')
    print(csv_list)

    ensemble_mean(csv_list)