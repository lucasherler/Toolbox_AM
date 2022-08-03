#Attribute Information:
#
#1. Sample code number: id number
#2. Clump Thickness: 1 - 10
#3. Uniformity of Cell Size: 1 - 10
#4. Uniformity of Cell Shape: 1 - 10
#5. Marginal Adhesion: 1 - 10
#6. Single Epithelial Cell Size: 1 - 10
#7. Bare Nuclei: 1 - 10
#8. Bland Chromatin: 1 - 10
#9. Normal Nucleoli: 1 - 10
#10. Mitoses: 1 - 10
#11. Class: (2 for benign, 4 for malignant)


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
        

def check_missing(x):
    f = open(x,'r')
    idx = []
    line = f.readline()
    k=0
    while line:
        if '?' in line:
            idx.append(k)
        k = k+1
        line = f.readline()    
    return idx

def read():
    # print("\n LEU IRIS")
    
    idx = check_missing('breast-cancer-wisconsin.data')
    df = pd.read_csv('breast-cancer-wisconsin.data',\
                     names=['id','clumpthick', 'unifcellsize','unifcellshape', \
                            'marginaladh','singleepith','barenuclei', \
                            'blandchromatin','normalnucleoli','mitoses','class'],skiprows=idx)
    # df['Class'] = df['Class'].transform(transform)
    df = df.to_numpy()
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded = onehot_encoder.fit_transform(df[:,8])
    return (df[:,1:-1].astype(float),one_hot_encoding(df[:,-1]))

def is_regression():
    return False


if __name__ == '__main__':
    df = read()