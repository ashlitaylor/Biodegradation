from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        #* Initializing the tree as an empty dictionary
        self.tree = {}
        pass

    def learn(self, X, y):
        #Train the decision tree (self.tree) using the the sample X and labels y
        #The functions in utils.py are used to train the tree
        
        #* Method used to implement the tree:
        #*    Each node in self.tree is in the form of a dictionary:
        #*       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #*    For example, a non-leaf node with two children can have a 'left' key and  a 
        #*    'right' key. 
        leny = len(set(y))
        #print(leny)
        if leny == 1:  
            self.tree['label'] = y[0]
            return
        if leny == 0:
            self.tree['label'] = 0
            return        
        
        x_left = []
        x_right = []
        y_left = []
        y_right = []

        maxInfoGain = 0
        maxIndex = 0
        best_split_attr = 0
        split_val = 0
        
        terminate_case = 0, 0.0, 0

        for i in range(len(X)-4):
        
                for j in range(len(X[0])):
                    split_val_update = X[i][j]
                    x_l, x_r, y_l, y_r = partition_classes(X,y,j,split_val_update)
                    y_update = [y_l, y_r]
                    infoGain = information_gain(y, y_update)
                    if infoGain > maxInfoGain: 
             
                            maxInfoGain = infoGain
                            split_val = split_val_update
                            best_split_attr = j
                            testing = i, maxInfoGain, best_split_attr

                            x_left = x_l
                            x_right = x_r
                            y_left = y_l
                            y_right = y_r

        
        self.tree['left'] = DecisionTree()
        self.tree['right'] = DecisionTree()
        self.tree['left'].learn(x_left,y_left)
        self.tree['right'].learn(x_right, y_right)
        self.tree['attribute'] = best_split_attr
        self.tree['value'] = split_val

        pass


    def classify(self, record):
        #Classify the record using self.tree and return the predicted label
        root = self.tree
        
        while 'value' in root:
            split_attribute = root['attribute']
            split_val = root['value']

            if isinstance(record[split_attribute], int):
                if record[split_attribute] <= split_val:
                    root = root['left'].tree
                else:
                    root = root['right'].tree
            else:
                if record[split_attribute] == split_val:
                    root = root['left'].tree
                else:
                    root = root['right'].tree

        return root['label']
        
        pass
