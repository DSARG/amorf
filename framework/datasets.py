import numpy as np 
import arff

#Targets 16, Attributes 14 , Instances 
def load_WQ():
    file = open('wq.arff')
    dataset = arff.load(file)
    data = np.array(dataset['data'])
    X = data[:,16:30] 
    y = data[:,0:16]  
    file.close()
    return X,y 

#Targets 16    
def load_RF1(): 
    file = open('rf1.arff')
    dataset = arff.load(file) 
    data = np.array(dataset['data'])
    X = data[:,0:64]
    y = data[:,64:72] 
    X[X == None] = 0 
    y[y == None] = 0
    file.close()
    return X.astype(np.float32),y.astype(np.float32) 

def load_EDM(): 
    file = open('edm.arff')
    dataset = arff.load(file) 
    data = np.array(dataset['data'])
    X = data[:,0:16]
    y = data[:,16:18]
    file.close()
    return X.astype(np.float32),y.astype(np.float32) 

''' TODO:   Forestry-Kras (large)
            Vegetation Clustering (large)
            Wine Quality?
            '''
