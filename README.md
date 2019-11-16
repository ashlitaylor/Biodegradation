-------------------------------------------------------
# Random Forest Classifier on the QSAR Biodegradation Data Set
-------------------------------------------------------
The python code was completed as part of an assignment for my Data and Visual Analytics course at The Georgia Insitute of Technology. 

### Introduction
The Quantitative Structure Activity Relationships (QSAR) Bidegradation [dataset](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) was collected to facilitate the study of the relationships between chemical structure and biodegradation of molecules. Models based on QSAR provide promise a relatively inexpensive means of determining the biodegradability of chemicals. The desire for this kind of testing stems from a desire around the world to reduce the amount of non-biodegradable matrials as one method of improving sustainability.  

The dataset has 42 columns: the first 41 columns represent the attributes of the molecule, and the last column is the ground ­truth label which tells us if the molecule is biodegradable or not (1 means biodegradable, 0 means not biodegradable). The dataset has been cleaned to remove missing attributes, and is comprised of 1055 instances. 

##### OOB Error Estimate
This python code builds a random forest from scratch and performs classification on the QSAR dataset. Instead of cross validation, I used  the Out-of-bag (OOB) error estimate to evaluate the model performance. Each tree in the forest is constructed using a different bootstrap sample from the original data. Each bootstrap sample is constructed by randomly sampling from the original dataset with replacement. The proportion of times that a predicted class is not equal to the true class of a record averaged over all records is the OOB error estimate. 

Bootstrapping allows confidence intervals to be placed on the model predictions based on uncertainties in the input data. Being able to obtain the uncertainty level of individual model predictions for chemical biodegradability is desireable as public and private entities pursue regulatory and chemical design goals. 

This code is comprised of three files: 

1. A util.py code file that contains utility functions to build the decision tree. Implements the functions to compute entropy, information gain, and perform splitting.
2. A decision_tree.py code file that contains a decision tree class to build the tree. Implements the classify() method to predict the label of a test record and the learn() method to build the decision tree using the utility functions above. 
3. A random_forest.py code file that contains a random forest class and a main method to test the random forest. Implements the functions _bootstrapping(), fitting(), voting().

The Random Forest uses entropy gain to select the splitting attribute and split point for the selected attribute and achieves 80% accuracy on the QSAR dataset. 

### Prerequisites
This code was writting in Python 3.7, and requires the following four files to be in the same folder/directory to run:
* util.py
* decision_tree.py
* random_forest.py
* qsardata.csv

### Run
I used Spyder to run this code, however it can be ran from your favorite Python IDE. Simply run the random_forest.py code, and it will output the model results for the qsardata.csv file. 


