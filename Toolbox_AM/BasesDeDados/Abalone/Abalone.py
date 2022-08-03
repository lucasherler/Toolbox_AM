import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# 
#Name / Data Type / Measurement Unit / Description
#-----------------------------
#Sex / nominal / -- / M, F, and I (infant)
#Length / continuous / mm / Longest shell measurement
#Diameter / continuous / mm / perpendicular to length
#Height / continuous / mm / with meat in shell
#Whole weight / continuous / grams / whole abalone
#Shucked weight / continuous / grams / weight of meat
#Viscera weight / continuous / grams / gut weight (after bleeding)
#Shell weight / continuous / grams / after being dried
#Rings / integer / -- / +1.5 gives the age in years

#The readme file contains attribute statistics.

def transform(x):
    if x == 'M':
        x = 1
    elif x == 'F':
        x = -1
    else:
        x = 0 
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
    df = pd.read_csv('abalone.data',names=['Sex','Length','Diameter','Height', \
                                           'Whole weight','Shucked Weight', \
                                           'Viscera Weight', 'Sheel Weight','Rings'])
    df['Sex'] = df['Sex'].transform(transform)
    df = df.to_numpy()
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded = onehot_encoder.fit_transform(df[:,8])
    return (df[:,0:8].astype(float),one_hot_encoding(df[:,8]))

def is_regression():
    return False


if __name__ == '__main__':
    df = read()