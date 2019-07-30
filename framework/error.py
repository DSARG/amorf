from numpy import mean

def average_relative_error(y_test,y_pred):
    dim = len(y_pred[0,:])
    result = 0
    for i in range(0,dim): 
        sum_of_relative_error = 0
        for j in range(0, len(y_test)):
            sum_of_relative_error += abs(y_test[j,i]-y_pred[j,i])/abs(y_test[j,i]) #not sure wether to use abs()here
        avg_sre = sum_of_relative_error/len(y_test) 
        result += avg_sre 
    return result/dim

def average__relative_root_mean_squared_error(y_test,y_pred): 
    dim = len(y_pred[0,:])
    result = 0
    for i in range(0,dim):
        sum_squared_error = 0
        for j in range(0, len(y_test)): 
            sum_squared_error += ((y_test[j,i]-y_pred[j,i])**2)
        sum_squared_deviation = 0

        for k in range(0, len(y_test)):
            sum_squared_deviation += (y_test[k,i]- mean(y_test[:,i]))**2
        
        result += (sum_squared_error/sum_squared_deviation)**(1/2) 
    return result/dim