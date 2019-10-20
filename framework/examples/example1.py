from framework.datasets import EDM

data_X, data_y = EDM.EDM().get_numpy()
print(data_X[0])
print(data_y[0])
