# Outline Presentation 09.09.19

## Contents 
0. *Background Story* ?
1. Motivation 
2. Goal 
3. Current Status 
* Methods 
    * Single Target Methods 
    * Regression Trees 
    * Neuronal Nets 
    * Multi-Output Least Squares Supprt Vector Regression 
* Datasets 
5. Presentation/Dash Prototype 
5. Shedule/Milestones 

---
## 1. Motivation 
* Definition of Problem, Terminology, Multi-Variate, Multi-Target, Multi-Output 
* https://onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1157
## 2. Goal  
Construction of a Python package that includes several methods for multi-output regression. The Framework will be shipped via pip, it will contain a front-end application for ease of use. The Methods include Single Target Methods, Multi-Output Regression Tree, at least two types of Neuronal Networks and an implementation of a Multi-Output Support Vector Regression Method. The Code shall *largely* be finished before Christmas. The Thesis shall be finished two weeks before due date.  

*Mention Focus on Software Development*

## 3. Current Status 
### Overview of Project structure  


### Methods  
#### Single Target Methods  
* Formulation of Method 
* List of currently supported Estimators   
    * linear
    * kneighbors
    * adaboost
    * gradientboost
    * mlp
    * svr
    * xgb 
* https://www.researchgate.net/publication/263813673_A_Review_On_Multi-Label_Learning_Algorithms - useless 
* https://www.semanticscholar.org/paper/Multi-Label-Classification-Methods-for-Multi-Target-Xioufis-Groves/5ae7139ba40efeee40c2d6ac22666370b91ce77d - see 3.0

#### Regression Trees  
* https://towardsdatascience.com/https-medium-com-lorrli-classification-and-regression-analysis-with-decision-trees-c43cdbc58054 
* https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/ 
#### Neuronal Nets  
* Structure of the Nets  
* Structure of Code (Estimator that works with different models)
* Features of the estimator (early Stopping, validation, different ooptimizers) 
* Adam,SGD - maybe pseudocode
* Guide https://medium.com/inbrowserai/simple-diagrams-of-convoluted-neural-networks-39c097d2925b
* Any Formalities? https://discuss.pytorch.org/t/graph-visualization/1558/2 
* http://alexlenail.me/NN-SVG/index.html For FC 
* https://www.draw.io/ For CNN  

##### Linear Network? (Maybe prepare slides, at least understand) 
* ReLU - https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7 
* DropOut - https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
* BatchNorm - https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

#### Multi-Output Least Squares Supprt Vector Regression 
* Problems, Ideas for improvement(Dask. NumExpr, NP.Memmap) 

### Datasets 
* Table of Current/Planned Datasets and their Structure  

### Error Metrics 
* RMSE - Problems 
* ARRMSE - Problems  
* https://stats.stackexchange.com/questions/260615/what-is-the-difference-between-rrmse-and-rmsre  
* https://www.sciencedirect.com/science/article/pii/S1364032115013258?via%3Dihub

## 4. Presentation/Dash Prototype 
* If already developed a short interactive presentation of the current status 
* Otherwise Mockups of UI and a Test in Terminal

## 5. Shedule/Milestones 
(Gantt-Chart or Timeline)
* Thesis Registration  
* Implementation Goals  
...
* Acquisition of Datasets 
* Acquisition of Literature Sources
* Acquisition of Licenses
* Completion of Code Base (*around Christmas*)
* Thesis Chapter Goals  
... 
* Completion of Thesis  
(*two weeks*)
* Deadline 

## 6. Structure of Thesis 
