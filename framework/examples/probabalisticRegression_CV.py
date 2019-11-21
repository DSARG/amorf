import framework.datasets as ds
import framework.probabalisticRegression as probabalisticRegression
import framework.metrics as metrics
import numpy as np
from sklearn.model_selection import KFold 

edm = ds.EDM().get_numpy()
rf1 = ds.RiverFlow1().get_numpy()
wq = ds.WaterQuality().get_numpy()
transCond = ds.TransparentConductors().get_numpy()
dataset_names = ['EDM', 'RF1', 'Water Quality', 'Transparent Conductors']
datasets = [edm, rf1, wq, transCond]
results_datasets = [] 
for dataset in datasets: 
    all_results = [] 
    X = dataset[0]
    y = dataset[1] 
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    selector_results = [] 
    for train_index, test_index in kf.split(X): 
        BNN = probabalisticRegression.BayesianNeuralNetworkRegression(patience=6,learning_rate=1e-5) 
        fitted = BNN.fit( X[train_index], y[train_index])
        prediction, prediction_stds = fitted.predict(X[test_index],y[test_index]) 
        result = metrics.average_relative_root_mean_squared_error(
            prediction, y[test_index])
        selector_results.append(result) 
    all_results.append(selector_results) 
    means_and_std = []
    for result in all_results:
        mean, std = np.mean(result), np.std(result)
        means_and_std.append([mean, std])

    results_datasets.append(means_and_std) 
dataset_counter = 0
output = ""
for dataset in results_datasets:
    result_counter = 0
    print(dataset_names[dataset_counter])
    output += dataset_names[dataset_counter] + '\n\n'
    dataset_counter += 1
    for selector in dataset:
        print("Mean\t\t\tStd Dev\n {} \t {}".format(selector[0], selector[1]))
        output += "Mean\t\t\tStd Dev\n {} \t {}\n".format(
            selector[0], selector[1])
        result_counter += 1

with open("ProbabalisticRegression_CV.txt", "w") as text_file:
    text_file.write(output) 