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
    # sed = np.loadtxt('phoneme.dat', comments=['@'],delimiter=',', converters={0: lambda x: x}) 
    # sed = np.genfromtxt('satimage.scale', delimiter=' ') 
    # f = open('segment!.scale','r+')
    # count = 0;
    # line = f.readline()
    # while line:
    #     count = count +1;
    #     line = f.readline()
    # xx = np.zeros([count,19])
    # yy = np.zeros([count,1])
    # count = 0
    # f.seek(0)
    # line = f.readline()
    # while line:
    #     tokens = line.split(' ')
    #     yy[count] = float(tokens[0])
    #     for i in range(1,len(tokens)):
    #         if tokens[i] == '\n':
    #             continue
    #         a,b = tokens[i].split(':')
    #         xx[count,int(a)-1] = float(b)
    #     count = count+1;
    #     line = f.readline()
            
    # return (xx,one_hot_encoding(yy))
    sed = np.loadtxt('thyroid.dat', comments=['@'],delimiter=',', converters={0: lambda x: x}) 
    return (sed[:,:-1],one_hot_encoding(sed[:,-1]))

def is_regression():
    return False


if __name__ == '__main__':
    df = read()