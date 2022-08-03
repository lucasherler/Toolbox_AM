#In the data set there are 857 attributes, 1 attributes with the class of instance and 856 with word frequency:
#1. category: range 1 - 9 (integer)
#2 - 857. word frequency: (integer)


import numpy as np
import pandas as pd

def one_hot_encoding(x):
    values = np.unique(x).tolist()
    # print(type(values))
    encoded = np.zeros([x.shape[0],len(values)])
    for i in range(x.shape[0]):
        encoded[i,values.index(x[i])] = 1
    return encoded
        

def read():
    # print("\n LEU IRIS")
    df = pd.read_csv('CNAE-9.data',names=range(0,857))
    # df['Class'] = df['Class'].transform(transform)
    df = df.to_numpy()
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded = onehot_encoder.fit_transform(df[:,8])
    return (df[:,1:].astype(float),one_hot_encoding(df[:,0]))

def is_regression():
    return False


if __name__ == '__main__':
    df = read()