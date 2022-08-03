# import os

import numpy as np

def one_hot_encoding(x):
    values = np.unique(x).tolist()
    # print(type(values))
    encoded = np.zeros([x.shape[0],len(values)])
    for i in range(x.shape[0]):
        encoded[i,values.index(x[i])] = 1
    return encoded

# sed = np.loadtxt('coil2000.dat', comments=['@'],delimiter=',', converters={0: lambda x: x}) 


def read():
    # print("\n LEU IRIS")
    # df = pd.read_csv('abalone.data',names=['Sex','Length','Diameter','Height', \
    #                                        'Whole weight','Shucked Weight', \
    #                                        'Viscera Weight', 'Sheel Weight','Rings'])
    # df['Sex'] = df['Sex'].transform(transform)
    # df = df.to_numpy()
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded = onehot_encoder.fit_transform(df[:,8])
    sed = np.loadtxt('enb.arff', comments=['@'],delimiter=',', converters={0: lambda x: x}) 
    return (sed[:,:-2],sed[:,-2:])

def is_regression():
    return True


if __name__ == '__main__':
    df = read()