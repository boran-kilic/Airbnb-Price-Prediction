import numpy as np



##########################common functions#####################################
def r2_score (y_test, y_predict):
    nominator = 0
    denominator=0
    for i in range(len(y_test)):
        nominator = nominator + (y_test[i]-y_predict[i])**2
    for i in range(len(y_test)):
        denominator =  denominator + (y_test[i]- y_test.mean())**2
    return 1-(nominator/denominator)    

def calc_RSS (y_test, y_predict):
    RSS = 0
    for i in range(len(y_test)):
        RSS = RSS + (y_test[i]-y_predict[i])**2    
    return RSS

def mean_squared_error (y_test, y_predict):
    RSS = calc_RSS (y_test, y_predict)
    MSE = RSS/len(y_test)
    return MSE

def root_mean_squared_error(y_test,y_predict):
    MSE = mean_squared_error (y_test, y_predict)
    return np.sqrt(MSE)

def mean_absolute_error (y_test, y_predict):
    total = 0
    for i in range(len(y_test)):
        total = total + abs(y_test[i]-y_predict[i])
    MSE = total/len(y_test)
    return MSE

def train_test_split(x,y, seed, test_size, validation_size=0):
    np.random.seed(seed)  
    test_size = int(len(x) * test_size)     
    validation_size = int(len(x) * validation_size) 
    x = x.to_numpy()
    # y = y.to_numpy()
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    x_shffl = x[indices]
    y_shffl = y[indices]
    
    x_train = x_shffl[validation_size + test_size:]
    x_test = x_shffl[validation_size:validation_size + test_size]
    x_validation = x_shffl[:validation_size]
    
    y_train = y_shffl[validation_size + test_size:]
    y_test = y_shffl[validation_size:validation_size + test_size]
    y_validation = y_shffl[:validation_size]    
    
    if validation_size == 0:
        return x_train, x_test, y_train, y_test    
    else:
        return x_train, x_test, y_train, y_test, x_validation, y_validation


 













































