#Attribute Information:
#
#1. Sequence Name: Accession number for the SWISS-PROT database
#2. mcg: McGeoch's method for signal sequence recognition.
#3. gvh: von Heijne's method for signal sequence recognition.
#4. lip: von Heijne's Signal Peptidase II consensus sequence score. Binary attribute.
#5. chg: Presence of charge on N-terminus of predicted lipoproteins. Binary attribute.
#6. aac: score of discriminant analysis of the amino acid content of outer membrane and periplasmic proteins.
#7. alm1: score of the ALOM membrane spanning region prediction program.
#8. alm2: score of ALOM program after excluding putative cleavable signal regions from the sequence.


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
    df = pd.read_csv('ecoli.data',names=range(0,9),delimiter='\s+')
    # df['Class'] = df['Class'].transform(transform)
    df = df.drop(columns=0).to_numpy()

    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded = onehot_encoder.fit_transform(df[:,8])
    return (df[:,:-1].astype(float),one_hot_encoding(df[:,-1]))

def is_regression():
    return False


if __name__ == '__main__':
    df = read()