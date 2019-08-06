import numpy as np 
import arff

#Targets 16, Attributes 14 , Instances 
def load_WQ():
    dataset = arff.load(open('wq.arff'))
    data = np.array(dataset['data'])
    X = data[:,16:30] 
    y = data[:,0:16] 
    return X,y 
    
def load_RF1(): 
    dataset = arff.load(open('rf1.arff')) 
    data = np.array(dataset['data'])
    X = data[:,0:64]
    y = data[:,64:72] 
    X[X == None] = 0 
    y[y == None] = 0
    return X.astype(np.float32),y.astype(np.float32) 

def load_EDM(): 
    dataset = arff.load(open('edm.arff')) 
    data = np.array(dataset['data'])
    X = data[:,0:16]
    y = data[:,16:18]
    return X.astype(np.float32),y.astype(np.float32) 

''' TODO:   Forestry-Kras (large)
            Vegetation Clustering (large)
            Wine Quality?
            '''
