import amorf.datasets as ds
import amorf.probabalisticRegression as probabalisticRegression
import amorf.metrics as metrics
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


def perform_GridSearch(X, y):
    scorer = make_scorer(
        metrics.average_relative_root_mean_squared_error, greater_is_better=False)
    parameters = {
        "learning_rate": [0.01],# 0.001, 0.1, 0.2, 0.3],
        "patience": [1]#, 3, 5, 8]
    }

    reg = GridSearchCV(probabalisticRegression.BayesianNeuralNetworkRegression(
        training_limit=10000), parameters, cv=3, scoring=scorer, n_jobs=1)

    return reg.fit(X, y)


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
    BNN = perform_GridSearch(X, y).best_estimator_
    for train_index, test_index in kf.split(X):
        fitted = BNN.fit(X[train_index], y[train_index])
        prediction, prediction_stds = fitted.predict(
            X[test_index], y[test_index])
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
