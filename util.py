from scipy import stats
import numpy as np
import math


#* This method computes entropy for information gain
def entropy(class_y):
    #* Input:            
    #*   class_y         : list of class labels (0's and 1's)
    
    #Compute the entropy for a list of classes
    #*
    #* Example:
    #*    entropy([0,0,0,1,1,1,1,1,1]) = 0.92
        
    zeroCount = 0
    oneCount = 0
    #length_y = len(class_y)
    
    if len(class_y) == 0:
        return 0
        
    zeroCount = class_y.count(0)
    oneCount = class_y.count(1)
   
    prob_zero = (1.0*zeroCount)/len(class_y)
    prob_one = (1.0*oneCount)/len(class_y)
    #Entropy: H = -sum(-prob(i)*log_2(prob(i)))
    if prob_zero == 0:
        entropy = (-1.0*prob_one*math.log(prob_one,2))
    elif prob_one == 0:
        entropy = (-1.0*prob_zero*math.log(prob_zero,2))
    else:
        entropy = (-1.0*prob_zero*math.log(prob_zero,2)) + (-1*prob_one*math.log(prob_one,2)) 
    return entropy


def partition_classes(X, y, split_attribute, split_val):
    #* Inputs:
    #*   X               : data containing all attributes
    #*   y               : labels
    #*   split_attribute : column index of the attribute to split on
    #*   split_val       : either a numerical or categorical value to divide the split_attribute
    
    #* TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    #* 
    #* First: check if the split attribute is numerical or categorical    
    #* If the split attribute is numeric, split_val should be a numerical value
    #* If the split attribute is categorical, split_val should be one of the categories.   
    #*
    #* Numeric Split Attribute:
    #*   Split the data X into two lists(X_left and X_right) where the first list has all
    #*   the rows where the split attribute is less than or equal to the split value, and the 
    #*   second list has all the rows where the split attribute is greater than the split 
    #*   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #*
    #* Categorical Split Attribute:
    #*   Split the data X into two lists(X_left and X_right) where the first list has all 
    #*   the rows where the split attribute is equal to the split value, and the second list
    #*   has all the rows where the split attribute is not equal to the split value.
    #*   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:
    
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    
    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.
    
    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.
    
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
              
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.
        
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
              
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]
               
    ''' 
    
    X_left = []
    X_right = []
    
    y_left = []
    y_right = []

    if isinstance(X[0][split_attribute], int):
        for index, i in enumerate(X):
            if i[split_attribute] <= split_val:
                X_left.append(i)
                y_left.append(y[index])
            else:
                X_right.append(i)
                y_right.append(y[index])  
    else:
        for index, i in enumerate(X):
            if i[split_attribute] == split_val:
                X_left.append(i)
                y_left.append(y[index])
            else:
                X_right.append(i)
                y_right.append(y[index])            
    #print(X_left)
    #print(y_left)
    #print(X_right)
    #print(y_right)
    return (X_left, X_right, y_left, y_right)

    
def information_gain(previous_y, current_y):
    #* Inputs:
    #*   previous_y: the distribution of original labels (0's and 1's)
    #*   current_y:  the distribution of labels after splitting based on a particular
    #*               split attribute and split value
    
    #* Compute and return the information gain from partitioning the previous_y labels
    #* into the current_y labels.
    #* Information gain calculated using entropy function.
    #* Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf
    
    """
    Example:
    
    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    
    info_gain = 0.45915
    """
    #Information Gain: IG = H - ((H_L*P_L) +(H_R*P_R))
    info_gain = 0
    current_H = 0.0
    previous_H = 1.0*entropy(previous_y)
    leny = len(previous_y)
    for m in current_y:
        lenm = len(m)
        if lenm > 0:
            current_H += (entropy(m) * ((1.0* lenm)/leny)) 
    info_gain += round((previous_H - current_H),5)
    return info_gain
 
#Test
'''
X = [[3, 'aa', 10],                 
         [1, 'bb', 22],                      
         [2, 'cc', 28],                      
         [5, 'bb', 32],                      
         [4, 'cc', 32]]                      
y = [1,
    1,
    0,
    0,
    1]
partition_classes(X, y, split_attribute = 0, split_val = 3)

partition_classes(X, y, split_attribute = 1, split_val = 'bb')

previous_y = [0,0,0,1,1,1]
current_y = [[0,0], [1,1,1,0]]
info_gain = 0
current_H = 0.0
previous_H = 1.0*entropy(previous_y)
leny = len(previous_y)
for m in current_y:
    lenm = len(m)
    if lenm > 0:
        current_H += (entropy(m) * float(lenm/leny)) 
info_gain += round((previous_H - current_H),5)
print(info_gain)   
entropy_test = round((entropy([0,0,0,1,1,1,1,1,1])),2)
print(entropy_test)
'''