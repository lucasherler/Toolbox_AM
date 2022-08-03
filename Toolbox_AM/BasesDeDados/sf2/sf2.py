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

def convert_1(x):
    # print(x)
    if x == 'A'.encode():
        return 0
    elif x == 'B'.encode():
        return 1
    elif x == 'C'.encode():
        return 2    
    elif x == 'D'.encode():
        return 3
    elif x == 'E'.encode():
        return 4
    elif x == 'F'.encode():
        return 5
    elif x == 'H'.encode():
        return 6
    else:
        return -1
    
def convert_2(x):
    if x == ' X'.encode():
        return 0
    elif x == ' R'.encode():
        return 1
    elif x == ' S'.encode():
        return 2    
    elif x == ' A'.encode():
        return 3
    elif x == ' H'.encode():
        return 4
    elif x == ' K'.encode():
        return 5
    return -1

def convert_3(x):
    if x == ' X'.encode():
        return 0
    elif x == ' O'.encode():
        return 1
    elif x == ' I'.encode():
        return 2    
    elif x == ' C'.encode():
        return 3
    return -1

    
# @attribute mod_zurich_class {A,B,C,D,E,F,H}
# @attribute largest_spot_size {X,R,S,A,H,K}
# @attribute spot_distribution {X,O,I,C}

def read():
    # print("\n LEU IRIS")
    # df = pd.read_csv('abalone.data',names=['Sex','Length','Diameter','Height', \
    #                                        'Whole weight','Shucked Weight', \
    #                                        'Viscera Weight', 'Sheel Weight','Rings'])
    # df['Sex'] = df['Sex'].transform(transform)
    # df = df.to_numpy()
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded = onehot_encoder.fit_transform(df[:,8])
    sed = np.loadtxt('sf2.arff', comments=['@'],delimiter=',', converters={0: convert_1, 1:convert_2, 2:convert_3}) 
    return (sed[:,:-3],sed[:,-3:])

def is_regression():
    return True


if __name__ == '__main__':
    df = read()