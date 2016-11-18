import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing


class ImportData:
    def __init__(self, vs: bool = False):
        """
        The initialisation step will input the data into memory (data is not that large easily fits
        into RAM)
        """
        if vs:
            path = 'C:/Users/Thomas/Documents/Data/Santander/train_ver2.csv'
        else:
            path = '~/Data/SantanderDataScience/train_ver2.csv'

        # Get the feature set
        self.dataset = pd.read_csv(path, nrows=100000)

        self.X = self.dataset.iloc[:, 0:24]

        # There seems to be 0.6% of rows that have NaN and this is ubiquitous among the feature set
        self.X = self.X[np.isfinite(self.X['indrel_1mes'])]
        # =======================================================================================
        #                                        CLEANING 
        # =======================================================================================
        # Do some cleaning of the categorical data to get dummy variables for these columns

        # Convert the dates to days from the beginning of the year or from now, whichever
        # makes more sense.
        # =======================================================================================
        # Convert the record date into days and set the start as the beginning of 2015
        self.X['days_recorded_from_ny'] = self.X.fecha_dato.apply(lambda x:
                                                         (datetime.datetime.strptime(x, "%Y-%m-%d") -
                                                          datetime.datetime(2015, 1, 1)).days)
        self.X.drop('fecha_dato', axis=1, inplace=True)

        # Convert the tenure date to days from now so it is a proper length. We will also need
        # to scale this later
        self.X['tenure_days_from_now'] = self.X.fecha_alta.apply(lambda x:
                                                   (datetime.datetime.now() -
                                                    datetime.datetime.strptime(x, "%Y-%m-%d")).days
                                                   if isinstance(x, str) else 0)
        self.X.drop('fecha_alta', axis=1, inplace=True)

        # Convert the primary customer churn date to a delta between date and now represented
        # as days. This will need to be scaled later.
        self.X['days_primary_churn_from_now'] = self.X.ult_fec_cli_1t.apply(lambda x:
                                       (datetime.datetime.now() -
                                        datetime.datetime.strptime(x, "%Y-%m-%d")).days
                                       if isinstance(x, str) else 0)
        self.X.drop('ult_fec_cli_1t', axis=1, inplace=True)

        # Drop the nomprov column as this is replicated in the cod_prov feature
        self.X.drop('nomprov', axis=1, inplace=True)

        # Turn sex into a dummy variable and concatenate it to the original feature set
        self.X['male'] = self.X['sexo'].apply(lambda x: 1 if 'H' else 0)
        self.X.drop('sexo', axis=1, inplace=True)

        # Clean the residence index set Nan to 'S'. We may have to delete the rows that are
        # associated with NaNs in this column because we are being forced to make a choice
        self.X['indresi'].fillna('S')
        self.X['resident'] = self.X['indresi'].apply(lambda x: 1 if 'S' else 0)
        self.X.drop('indresi', axis=1, inplace=True)

        # Clean the foreign index set NaN to 'N'. We make this assumption because we would
        # expect that most banking customers would be from spain and the proportion of NaNs
        # are low compared with the amount of foreigners. Hopefully this will be dimensionally
        # reduced when we run through the PCA analysis
        self.X['indext'].fillna('S')
        self.X['foreign'] = self.X['indext'].apply(lambda x: 1 if 'S' else 0)
        self.X.drop('indext', axis=1, inplace=True)

        # Set the new customer index to a binary value
        self.X['new_cust'] = self.X.innd_nuevo.fillna(0)

        # primary customer at the end of the month
        self.X['indrel'] = self.X.indrel.apply(lambda x: 1 if 1 else 0)

        # Change the employee spouse reference conyuemp to fill NaN with 0. This is then binary
        self.X.conyuemp.fillna(0)

        # =======================================================================================
        #                              SETTING THE DUMMY VARIABLES
        # =======================================================================================
        d_account_type = pd.get_dummies(self.X['indrel_1mes'])
        d_employee_ind = pd.get_dummies(self.X['ind_empleado'])
        d_customer_relation = pd.get_dummies(self.X['tiprel_1mes'])
        d_channel = pd.get_dummies(self.X['canal_entrada'])
        d_province = pd.get_dummies(self.X['cod_prov'])
        d_country_res = pd.get_dummies(self.X['pais_residencia'])
        d_class = pd.get_dummies(self.X['segmento'])

        # Attach all of the dummy variables to the feature set and delete the columns that
        # they came from
        self.X = pd.concat([self.X, d_employee_ind, d_account_type, d_customer_relation,
                   d_channel, d_province, d_country_res, d_class])

        # ======================================================================================
        #                            SCALING THE CONTINUOUS VARIABLES
        # ======================================================================================

        preprocessing.scale(self.X.age)

        self.Y = self.dataset.iloc[:, 25:-1]

    def return_list_of_cat_cols(self):
        for i in self.X.columns:
            print(i, type(self.X.loc[0][i]), len(self.X[i].unique()))

    def calculate_mem_consump(self):
        nbytes = self.X.values.nbytes + self.X.index.nbytes + self.X.columns.nbytes
        return nbytes / 1000000000

    def calculate_dummy_cols(self):
        pass
