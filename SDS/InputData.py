import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
import networkx as nx 
import matplotlib.pyplot as plt
import itertools
import math

class ImportData:
    def __init__(self, vs: bool = False, set_dummies: bool=False, all_rows: bool=True, scaling: bool=False):
        """
        The initialisation step will input the data into memory (data is not that large easily fits
        into RAM)
        """
        if vs:
            path = 'C:/Users/Thomas/Documents/Data/Santander/train_ver2.csv'
        else:
            path = '~/Data/SantanderDataScience/train_ver2.csv'

        # Get the feature set
        if all_rows:
            self.dataset = pd.read_csv(path)
        else:
            self.dataset = pd.read_csv(path, nrows=1000)

        self.X = self.dataset.iloc[:, 0:24]

        # There seems to be 0.6% of rows that have NaN and this is ubiquitous among the feature set
        self.X = self.X[(self.X['age']!='NaN') & (self.X['age']!= 'NA') & (self.X['age']!=' NA')]
        
        # Get rid of the customer code. We want to identify patterns in customer behaviour, I think that
        # matching on customer code will introduce some bias. We can always revisit this later.  
        self.X.drop('ncodpers', axis=1, inplace=True)
       
        # =======================================================================================
        #                                        CLEANING 
        # =======================================================================================
        # Do some cleaning of the categorical data to get dummy variables for these columns

        # Convert the dates to days from the beginning of the year or from now, whichever
        # makes more sense.
        # =======================================================================================
        # Earnings
        self.X.renta = self.X.renta.apply(lambda x: -99 if math.isnan(x) else x)
        self.earn_ind = np.where(self.X.renta==-99)

        # Convert string to continuous where necessary
        self.X['age'].apply(lambda x: np.float_(x))
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
        self.X['new_cust'] = self.X.ind_nuevo.fillna(0)

        # primary customer at the end of the month
        self.X['indrel'] = self.X.indrel.apply(lambda x: 1 if 1 else 0)

        # Change the employee spouse reference conyuemp to fill NaN with 0. This is then binary
        self.X.conyuemp.fillna(0)

        # The dummy variables for the indrel1_1mes and tiprel_1mes will not work correctly until the
        # following have been applied. This creates recognisable and standadised dummy variables
        self.X.indrel_1mes = self.X.indrel_1mes.apply(lambda x: self.run_f(str(x)[0]))
        self.X.tiprel_1mes = self.X.tiprel_1mes.apply(lambda x: self.run_f2(str(x)[0]))
        self.X.ind_empleado = self.X.ind_empleado.apply(lambda x: 'not_emp' if x=='N' else x)
        self.X.ind_empleado = self.X.ind_empleado.apply(lambda x: self.run_f3(x))
        self.X.segmento = self.X.segmento.apply(lambda x: 'none_type_work' if x=='NaN' else x)

        # =======================================================================================
        #                              SETTING THE DUMMY VARIABLES
        # =======================================================================================
        if set_dummies:
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
                       d_channel, d_province, d_country_res, d_class], axis=1)

            self.X.drop('indrel_1mes', axis=1, inplace=True)
            self.X.drop('ind_empleado', axis=1, inplace=True)
            self.X.drop('tiprel_1mes', axis=1, inplace=True)
            self.X.drop('canal_entrada', axis=1, inplace=True)
            self.X.drop('cod_prov', axis=1, inplace=True)
            self.X.drop('pais_residencia', axis=1, inplace=True)
            self.X.drop('segmento', axis=1, inplace=True)

        # ======================================================================================
        #                            SCALING THE CONTINUOUS VARIABLES
        # ======================================================================================
        if scaling:
            self.X.age = preprocessing.scale(self.X.age)
            self.X.days_recorded_from_ny = preprocessing.scale(self.X.days_recorded_from_ny)
            self.X.tenure_days_from_now = preprocessing.scale(self.X.tenure_days_from_now)
            self.X.antiguedad = preprocessing.scale(self.X.antiguedad)
            self.X.renta = preprocessing.scale(self.X.renta)
            self.X.renta[self.earn_ind[0]] = -2

        #Set the target set
        self.Y = self.dataset.iloc[:, 25:-1]

    def return_list_of_cat_cols(self):
        for i in self.X.columns:
            print(i, type(self.X.loc[0][i]), len(self.X[i].unique()))

    def calculate_mem_consump(self):
        nbytes = self.X.values.nbytes + self.X.index.nbytes + self.X.columns.nbytes
        return nbytes / 1000000000

    def calculate_dummy_cols(self):
        pass

    def run_f(self, input):
        x = ''
        if input=='1':
            x = 'primary_type'
        elif input=='2':
            x = 'coowner_type'
        elif input=='3':
            x = 'former_primary_type'
        elif input=='4':
            x = 'former_coowner_type'
        elif input=='P':
            x = 'potential_type'
        else:
            x= 'none_type'
        return x
    
    def run_f2(self,input):
        x = ''
        if input=='I':
            x = 'inactive_rel'
        elif input=='A':
            x = 'active_rel'
        elif input=='P':
            x = 'former_rel'
        elif input=='R':
            x = 'potential_rel'
        else:
            x= 'none_rel'
        return x

    def run_f3(self, input):
        x = ''
        if input=='not_emp':
            x = 'not_emp'
        elif input=='B':
            x='ex_emp'
        elif input=='F':
            x='filial_emp'
        elif input=='A':
            x='active_emp'
        elif input=='P' or input=='S':
            x='pass_emp'
        else:
            x='none_type_emp'
        return x
    
    def create_adjacency_of_products(self):
        l = self.Y.shape[1]
        A = np.repeat(0, l**2).reshape(l, l)
        for i,r in self.Y.iterrows():
            if sum(r)>1:  
                for j,k in itertools.combinations(np.where(r)[0], 2):
                    A[k][j] = A[k][j] + 1 
                    A[k][k] = A[k][k] + 1

        for i in range(A.shape[0]):
            for j in range(i, A.shape[1]):
                A[i][j] = A[j][i]
        return A
    
    def plot_adj(self, A):
        df = pd.DataFrame(A, index = self.Y.columns.values, columns = self.Y.columns.values)
        fig1 = plt.figure(figsize=(8,8))
        plt.pcolor(df,cmap=plt.cm.Reds)
        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation='vertical') 
        plt.title('Product relationships visualisation')
        fig1.savefig('prod_adj.png')