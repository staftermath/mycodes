#!/usr/bin/env python
print "Loading libraries..."

import ProgressBar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Kaggle Competition: BNP Paribas Cardif Claims Management
URL: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
train: train.csv
test: test.csv
sample submission: sample_submission.csv
"""


__author__ = "Weiwen Gu"
__copyright__ = ""
__credits__ = ["Weiwen Gu"]
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Weiwen Gu"
__email__ = "gwengww@gmail.com"
__status__ = "Production"


class DataModel:
    """
    Store models used in project process
    """
    def __init__(self):
        pass


class DataStruct:
    """
    Store data used or generated in project process
    """
    def __init__(self, input):
        self.input = None
        self.columns = None
        with open(input, 'rb') as inputfile:
            self.input = pd.read_csv(inputfile)
        self.missing_treatment = pd.DataFrame(columns=self.input.columns, data=[[None]*len(self.input.columns)],
                                              index=["value"])
        pass

    def finalize(self):
        pass


class BNP:
    def __init__(self,traincsv,testcsv,path):
        self.train = DataStruct(path+traincsv)
        self.test = DataStruct(path+testcsv)

    def analyze(self,col):
        """
        col = str or list. If str, it must be a column name. If list, it must be a list of column name
        """
        if type(col) == str:
            if col not in self.train.input.columns:
                print "Invalid colname."
            elif col == "ID":
                print "ID: total count = {}".format(len(self.train.input))
            else:
                dtype = self.train.input.dtypes[col]
                missing_num = sum(self.train.input[col].isnull())
                title = "col name: " + col + " dtype: {}".format(dtype) +\
                        " | Missing = {0} ({1:.2f}%)".format(missing_num,missing_num*1.0/len(self.train.input[col]))
                if dtype in [np.int, np.int32, np.int64, np.float]:
                    unique_num = len(np.unique(self.train.input[col]))
                    if unique_num < 30:
                        print "Column {0} looks like Categorical. Number of unique values is {1}".format(col, unique_num)
                        self.train.input.hist(column=col)
                        plt.title(title)
                    else:
                        print "Column {0} looks like continuous variable.".format(col)
                        self.train.input.hist(column=col)
                        plt.title(title)
                    plt.show()
                elif dtype == np.object:
                    print "Column {0} looks like Categorical variable.".format(col)
                    counts = self.train.input.groupby(col).apply(lambda x:len(x))
                    print counts
                    plt.bar(range(len(counts.index)), counts)
                    plt.xticks(range(len(counts.index)), counts.index)
                    plt.title(title)
                    plt.show()
                else:
                    print dtype
        elif type(col) == list:
            for c in col:
                self.analyze(c)
        else:
            print "Wrong Type of Input"

    def null_treatment(self, col, treatment):
        """
        NaN/Null data value treatment:
        treatment: ["mean","newlevel","equalfreq"]
                                "mean": use train data average of all non missing value
                                "newlevel": create a new categorical level for missing value
                                "equalfreq": set all missing value equal to a know value whose target distribution is the same as the missing value

        """
        if treatment == "mean":
            self.train.missing_treatment[col] = np.mean(self.train.input[col])
            self.train.input[col].fillna(self.train.missing_treatment[col][0],inplace=True)
        elif treatment == "newlevel":
            if self.train.input[col].dtype == np.object:
                self.train.missing_treatment[col] = "Missing"
                self.train.input[col].fillna(self.train.missing_treatment[col][0],inplace=True)
            else:
                self.train.missing_treatment[col] = np.max(self.train.input[col])+1
                self.train.input[col].fillna(self.train.missing_treatment[col][0],inplace=True)
        elif treatment == "equalfreq":
            null_in_col = self.train.input[self.train.input[col].isnull()]
            null = 1.0*sum(null_in_col['target'])/len(null_in_col)
            freq_of_all_category = self.train.input.groupby(col).apply(lambda x: np.sum(x['target'])*1.0/len(x))
            distance = np.abs(freq_of_all_category - null)
            index_closest_category = freq_of_all_category[distance == min(distance)].index[0]
            self.train.missing_treatment[col] = freq_of_all_category[index_closest_category]
            print "In column {1}, missing values have {0:.2f} % Positive.".format(null*100, col)
            print "The closest value is {}".format(freq_of_all_category[index_closest_category])
            self.train.input[col].fillna(self.train.missing_treatment[col][0],inplace=True)


if __name__ == "__main__":
    main = BNP(traincsv=r"train.csv",testcsv="test.csv",path = r"H:\\kaggle\\BNP\\")
    print main.train.input.columns
    print sum(main.train.input['v3'].isnull())
    main.analyze(col='v3')
    main.null_treatment(col='v3', treatment='newlevel')
    print sum(main.train.input['v3'].isnull())
    main.analyze(col='v3')


