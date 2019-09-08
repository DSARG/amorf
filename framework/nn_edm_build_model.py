import datasets as ds 
import error as er
import multiOutputRegressors as mor 
from sklearn.model_selection import train_test_split
import random

X,y = ds.load_RF1()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) 
keeprunning = True
count = 0  

bestModel = None 
bestRes = 999999
while(keeprunning):
    print('Linear NN {}'.format(count)) 
    count += 1 
    model = mor.ConvNet(input_dim=len(X_train[0]),output_dim=len(y_train[0])) 
    pat = random.randint(1,4) 
    lr = 1*10 ** -random.randint(1,3)
    reg = mor.NeuronalNetRegressor(patience=pat,model = model,learning_rate=0.0001,print_after_epochs=1000,batch_size=1000)
    res = er.average__relative_root_mean_squared_error(y_test,reg.fit(X_train, y_train).predict(X_test))
    print(res)  
    print('rate: {}'.format(lr)) 
    print('patience: {}'.format(pat)) 
    if res < bestRes:
        bestRes = res 
        bestModel = model
    if count > 50:
        keeprunning = False 

print(bestRes)
reg = mor.NeuronalNetRegressor(patience=pat,model = bestModel,learning_rate=0.0001,print_after_epochs=1000)    
reg.save("R_Linear_rf1_best")  