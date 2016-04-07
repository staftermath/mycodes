#!/usr/bin/env python
print "Loading libraries..."

import ProgressBar
import itertools
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from sklearn.preprocessing import OneHotEncoder

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
    def __init__(self, clf, param={}):
        """
        clf: model function
        param: dictionary containing model parameter and values
        """
        self.model = clf(**param)
        if hasattr(clf,"predict"):
            def predict(self,x):
                self.model.predict(x)

        if hasattr(clf,"transform"):
            def transform(self,x):
                self.model.transform(x)

    def fit(self,x,y=None):
        self.model.fit(x,y)




class DataStruct:
    """
    Store data used or generated in project process
    """
    def __init__(self, input):
        self.input = None
        self.columns = None
        with open(input, 'rb') as inputfile:
            self.input = pd.read_csv(inputfile)
        self.missing_treatment = {}
        pass

    def finalize(self):
        pass


class BNP:
    def __init__(self,traincsv,testcsv,path):
        self.train = DataStruct(path+traincsv)
        self.test = DataStruct(path+testcsv)
        self.xgboost = None
        self.onehotencoder = one_hot_encoder_batch()

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

    def null_treatment(self, col, function='train',treatment=None):
        """
        NaN/Null data value treatment:
        function: 'train': use train data set to generate null imputation value, store them in self.train.missing_treatment
                  'transform': use stored missing value treatment and imputation value to transform test data.
        treatment: ["mean","newlevel","equalfreq"]
                                "mean": use train data average of all non missing value
                                "newlevel": create a new categorical level for missing value
                                "equalfreq": set all missing value equal to a know value whose target distribution is the same as the missing value

        """
        def most_common(L):
            # get an iterable of (item, iterable) pairs
            SL = sorted((x, i) for i, x in enumerate(L))
            # print 'SL:', SL
            groups = itertools.groupby(SL, key=operator.itemgetter(0))
            # auxiliary function to get "quality" for an item

            def _auxfun(g):
                item, iterable = g
                count = 0
                min_index = len(L)
                for _, where in iterable:
                    count += 1
                    min_index = min(min_index, where)
                # print 'item %r, count %r, minind %r' % (item, count, min_index)
                return count, -min_index
            # pick the highest-count/earliest item
            return max(groups, key=_auxfun)[0]

        if function == "train":
            if treatment == "mean":
                self.train.missing_treatment[col] = ["mean",np.mean(self.train.input[col])]
                self.train.input[col].fillna(self.train.missing_treatment[col][1],inplace=True)
            elif treatment == "newlevel":
                if self.train.input[col].dtype == np.object:
                    self.train.missing_treatment[col] = ["newlevel","Missing"]
                    self.train.input[col].fillna(self.train.missing_treatment[col][1],inplace=True)
                else:
                    self.train.missing_treatment[col] = ["newlevel",np.max(self.train.input[col])+1]
                    self.train.input[col].fillna(self.train.missing_treatment[col][1],inplace=True)
            elif treatment == "equalfreq":
                null_in_col = self.train.input[self.train.input[col].isnull()]
                if len(null_in_col) == 0:
                    print "No missing value in column: {}".format(col)
                    self.train.missing_treatment[col] = ["most common",most_common(self.train.input[col])]
                else:
                    null = 1.0*sum(null_in_col['target'])/len(null_in_col)
                    freq_of_all_category = self.train.input.groupby(col).apply(lambda x: np.sum(x['target'])*1.0/len(x))
                    distance = np.abs(freq_of_all_category - null)
                    index_closest_category = freq_of_all_category[distance == min(distance)].index[0]
                    self.train.missing_treatment[col] = ["equalfreq",freq_of_all_category[index_closest_category]]
                    print "In column {1}, missing values have {0:.2f} % Positive.".format(null*100, col)
                    print "The closest value is {}".format(freq_of_all_category[index_closest_category])
                    self.train.input[col].fillna(self.train.missing_treatment[col][1],inplace=True)
            else:
                print "To train missing value imputation, treatment option is needed. Use \"mean\",\"newlevel\" or \"equalfreq\""

        elif function=="transform":
            if not self.train.missing_treatment:
                print "Missing value has not been imputated. Use function = \"train\" first"
            else:
                if col in self.train.missing_treatment:
                    self.test.input[col].fillna(self.train.missing_treatment[col][1],inplace=True)
                else:
                    print "Imputation was not performed for column: {}".format(col)
        else:
            print "Invalid Value for function. Use \"train\" or \"transform\""

    def xgboost_init(self,clf,param,reset = False):
        if not self.xgboost:
            self.xgboost = DataModel(clf,param)
        else:
            if not reset:
                print "XGB has already been initiated. Use reset = True to overwrite"
                print type(self.xgboost)
            else:
                self.xgboost = DataModel(clf,param)

    def generate_dummy_variables(self,col,remove_orig=True):
        """
        Use One-Hot Encoding to generate dummy variables for col. Naming the new variables as col_0,col_1,col_2...
        col: col name
        remove_orig: Remove the original col if True.
        """
        self.onehotencoder.fit(self.train.input[col])

class one_hot_encoder_batch:
    def __init__(self):
        """
        null_mapping: dict containing mapping values. it is initiated in fit step to be usually bnp.train.missing_treatment
        columns: list of cols that apply one_hot_encoder
        categorical_mapping: Dict stores one_hot_encoder fitted column and their corresponding level mapping.
        models: Dict stores col and its corresponding one_hot_encoder
        """
        self.null_mapping = None
        self.categorical_mapping = {}
        self.models = {}
        self.columns = None
    def _smart_mapping(self,col,df):
        res = []
        for x in df[col]:
            if x in self.categorical_mapping[col]:
                res.append(self.categorical_mapping[col][x])
            else:
                # if col in self.null_mapping:
                res.append(self.categorical_mapping[col][self.null_mapping[col][1]])
                # else:
        return np.array(res).reshape(len(res),1)


    def fit(self, list_of_col, null_mapping, df=None):
        """
        list_of_col: list of cols that apply one_hot_encoder
        df: DataFrame that contains cols
        """
        self.null_mapping = null_mapping
        self.columns = list_of_col
        for col in list_of_col:
            unique = np.unique(df[col])
            self.categorical_mapping[col]=dict(zip(unique,range(len(unique))))
            ohe = OneHotEncoder()
            num_col = self._smart_mapping(col,df)
            # num_col = np.array(map(lambda x:self.categorical_mapping[col][x], df[col])).reshape(len(df[col]),1)
            self.models[col] = ohe.fit(num_col)

    def transform(self,list_of_col = None, df=None):
        """
        transform list of col using fitted one_hot_encoder
        list_of_col: col name list, transform all stored fitted col from df
        rtype: DataFrame
        """
        res = pd.DataFrame(index=df.index)
        if not list_of_col or type(list_of_col) == list:
            columns = self.columns if not list_of_col else list_of_col
            for col in columns:
                num_col = self._smart_mapping(col,df)
                # num_col = np.array(map(lambda x:self.categorical_mapping[col][x], df[col])).reshape(len(df[col]),1)
                ohe_transformed = pd.DataFrame(data=self.models[col].transform(num_col).toarray(),
                                               columns=[col+'_'+str(n) for n in self.models[col].active_features_],index=df.index)
                res = res.merge(ohe_transformed, left_index=True, right_index=True)
                del ohe_transformed
        elif type(list_of_col) != list and type(list_of_col) != str:
            print "Invalid input for list_of_col. Use list of str or a str"
        else:
            col = list_of_col
            num_col = np.array(map(lambda x:self.categorical_mapping[col][x], df[col])).reshape(len(df[col]),1)
            ohe_transformed = pd.DataFrame(data=self.models[col].transform(num_col).toarray(),
                                           columns=[col+'_'+str(n) for n in self.models[col].active_features_], index=df.index)
            res = res.merge(ohe_transformed, left_index=True, right_index=True)
        return res

if __name__ == "__main__":
    main = BNP(traincsv=r"train.csv",testcsv="test.csv",path = r"C:\\kaggle\\BNP\\")
    # print main.train.input.columns
    # print sum(main.train.input['v3'].isnull())
    # main.analyze(col='v3')
    # main.null_treatment(col='v3', treatment='newlevel')
    # print sum(main.train.input['v3'].isnull())
    # main.analyze(col='v3')

    # for col in main.train.input.columns:
    #     if col != 'ID' and col != 'target':
    #         unique_element = len(np.unique(main.train.input[col]))
    #         if unique_element < 10:
    #             main.null_treatment(col=col,treatment='equalfreq')
    #         elif unique_element < 50:
    #             main.null_treatment(col=col,treatment='newlevel')
    #         else:
    #             main.null_treatment(col=col,treatment='mean')
    del main.train.input['v22']
    col_equalfreq = []
    col_mean = []
    col_newlevel = []

    # pbar = ProgressBar.ProgressBar(total=len(main.train.input.columns))
    col_need_to_transform = [col for col in main.train.input.columns if col != 'target' and col != 'ID']
    for col in col_need_to_transform:
        if main.train.input[col].dtypes == np.object:
            col_equalfreq.append(col)
            main.null_treatment(col=col, function="train", treatment='equalfreq')
        elif main.train.input[col].dtypes == np.float:
            col_mean.append(col)
            main.null_treatment(col=col, function="train", treatment='mean')
        elif main.train.input[col].dtypes in [np.int, np.int32, np.int64]:
            length = len(np.unique(main.train.input[col]))
            if length < 20 and length > 0:
                col_equalfreq.append(col)
                main.null_treatment(col=col, function="train", treatment='equalfreq')
            elif length < 200:
                col_newlevel.append(col)
                main.null_treatment(col=col, function="train", treatment='newlevel')
            else:
                col_mean.append(col)
                main.null_treatment(col=col, function="train", treatment='mean')
        else:
            print "Invalid type for column: {}".format(col)
        # pbar.update()

    del main.test.input['v22']

    print "Now Imputate Missing Values for test dataset"
    for col in col_need_to_transform:
        main.null_treatment(col=col, function="transform")

    print "Test has {0} cols. Train has {1} cols".format(len(main.test.input.columns),len(main.train.input.columns))
    print "Now Fitting One-Hot-Encoding for training data"
    main.onehotencoder.fit(col_equalfreq, null_mapping=main.train.missing_treatment, df=main.train.input)
    print "Now Getting arrays for One-Hot-Encoding for training Data"
    ohe_dataframe = main.onehotencoder.transform(df=main.train.input)
    for col in col_equalfreq:
        del main.train.input[col]
    main.train.input = main.train.input.merge(ohe_dataframe,left_index=True,right_index=True)

    print "Now Getting arrays for One-Hot-Encoding for Test Data"
    ohe_dataframe = main.onehotencoder.transform(df=main.test.input)
    for col in col_equalfreq:
        del main.test.input[col]
    main.test.input = main.test.input.merge(ohe_dataframe,left_index=True,right_index=True)

    print len(main.test.input.columns),len(main.train.input.columns)

    param = {"max_depth":3,"learning_rate":0.1,"n_estimators":300}
    main.xgboost_init(xgboost.XGBClassifier,param)
    feature = [col for col in main.train.input if col != 'ID' and col != 'target']
    x_train = main.train.input[feature]
    y_train = main.train.input['target']
    main.xgboost.fit(x_train,y_train)






