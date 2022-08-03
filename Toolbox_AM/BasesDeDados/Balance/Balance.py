#1. Class Name: 3 (L, B, R)
#2. Left-Weight: 5 (1, 2, 3, 4, 5)
#3. Left-Distance: 5 (1, 2, 3, 4, 5)
#4. Right-Weight: 5 (1, 2, 3, 4, 5)
#5. Right-Distance: 5 (1, 2, 3, 4, 5) 
import numpy as np
import pandas as pd


def transform(x):
    if x == 'L':
        x = 1
    elif x == 'B':
        x = 2
    else:
        x = 3 
    return x;

def one_hot_encoding(x):
    values = np.unique(x).tolist()
    # print(type(values))
    encoded = np.zeros([x.shape[0],len(values)])
    for i in range(x.shape[0]):
        encoded[i,values.index(x[i])] = 1
    return encoded
        

def read():
    # print("\n LEU IRIS")
    df = pd.read_csv('balance-scale.data',names=['Class','LeftWeight','LeftDist','RightWeight','RightDist'])
    df['Class'] = df['Class'].transform(transform)
    df = df.to_numpy()
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded = onehot_encoder.fit_transform(df[:,8])
    return (df[:,1:].astype(float),one_hot_encoding(df[:,0]))

def is_regression():
    return False


if __name__ == '__main__':
    df = read()