import numpy as np
import pandas as pd
import datetime.datetime

class import_data:
    def __init__(self, vs:bool=False):
        """
        The initialisation step will input the data into memory (data is not that large easily fits
        into RAM)
        """
        if vs:
            path = 'C:/Users/Thomas/Documents/Data/Santander/train_ver2.csv'
        else:
            path = '~/Data/SantanderDataScience/train_ver2.csv'

        #Get the feature set
        self.dataset = pd.read_csv(path)

        self.X = self.dataset.iloc[:, 0:23]
        #=======================================================================================
        #                                        CLEANING 
        #=======================================================================================
        # Do some cleaning of the categorical data to get dummy variables for these columns

        # Convert the dates to days from the beginning of the year
        # =======================================================================================
        self.X.fecha_dato_days = cls_dat.X.fecha_dato.apply(lambda x: 
                                                               (datetime.strptime(x, "%Y-%m-%d")-
                                                                datetime(2015,1,1)).days) 
        self.X.drop('fecha_dato', axis=1, inplace=True) 
        self.X.fetcha_alta_days = cls_dat.X.fetcha_alta.apply(lambda x: 
                                                               (datetime.strptime(x, "%Y-%m-%d")-
                                                                datetime(2015,1,1)).days)
        self.X.drop('fecha_alta', axis=1, inplace=True)
        # =======================================================================================

        # Change the employee spouse reference conyuemp to fill NaN with 0
        self.X.conyuemp.fillna(0)

        #d_col = pd.get_dummies(self.X[

        #self.Y = self.dataset.iloc[:, 24:-1]

    def return_list_of_cat_cols(self):
        for i in cls_dat.X.columns:
            print(i, type(cls_dat.X.loc[0][i]), len(cls_dat.X[i].unique()))

    def calculate_mem_consump(self):
        nbytes = self.X.values.nbytes + self.X.index.nbytes + self.X.columns.nbytes
        return nbytes/1000000000

    def calculate_dummy_cols(self):


