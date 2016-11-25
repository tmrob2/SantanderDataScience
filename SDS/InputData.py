import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
import networkx as nx 
import matplotlib.pyplot as plt
import itertools
import math

class ImportData:

    read_path = ''

    def get_path(self, ide):
        if ide == 'vs':
            path = 'C:/Users/Thomas/Documents/Data/Santander/train_ver2.csv'
        elif ide == 'spyder':
            path = 'C:/Users/611004435/Documents/Data/Santander/train_ver2.csv'
        else:
            path = '~/Data/SantanderDataScience/train_ver2.csv'
        return path

    def read_data(self, path):
        dataset = pd.DataFrame()
        if all_rows:
            dataset = pd.read_csv(path)
        else:
            if random_sample == False:
                dataset = pd.read_csv(path, nrows=10000)
            else:
                nlinesfile= 5000000
                nlines_rnd_sampl = 5000
                lines2skip = np.random.choice(np.arange(1,nlinesfile+1), 
                                              (nlinesfile-nlines_rnd_sampl), replace=False)
                for chunk in pd.read_csv(path, skiprows=lines2skip, chunksize=100000):
                    dataset = dataset.append(chunk)
        return dataset

    def read_X(self, dataset: pd.DataFrame, col_from:int, col_to:int):
        """ Allocate the feture set from the dataset"""
        X = self.dataset.iloc[:, col_from:col_to]
        return X

    def remove_nan_rows(self, X, col, ls):
        X = X[~X[col].isin(ls)]
        return X

    def drop_col(self, X, col_to_drop: list):
        for i in col_to_drop: 
            X.drop(i, axis=1, inplace=True)
        return X

    def apply_to_col(self, X, X_prime, new_col, f):
        X[new_col] = (X_prime).map(f)
        return X
    
    def clean_dataset(self, X):
        """
        =================================================================================
        Do some cleaning of the categorical data to get dummy variables for these columns

        Convert the dates to days from the beginning of the year or from now, whichever
        makes more sense.
        
        # ===============================================================================
        #Remove the rows without a tenure date
        """

        X = self.remove_nan_rows(X,'fecha_alta', ['NaN','NA',' NA']) 
        X['new_cust'] = X['ind_nuevo'].fillna(0)
        self.X.conyuemp = self.X.conyuemp.fillna(0)

        X = self.apply_to_col(X, X['age'].fillna(0), 'age', lambda x: np.float(x))
        X = self.apply_to_col(X, X['fecha_dato'], 'days_recorded_from_ny',
                                     lambda x: datetime.datetime.strptime(x,"%Y-%m-%d" )-
                                     datetime.datetime(2015,1,1))

        X = self.apply_to_col(X, X['fecha_alta'], 
                              'tenure', lambda x: (datetime.datetime.now() - 
                              datetime.datetime.strptime(x, "%Y-%m-%d")).days)

        X = self.apply_to_col(X, X['ult_fec_cli_1t'], 'days_churned', 
                              lambda x: (datetime.datetime.now() - 
                              datetime.datetime.strptime(x, "%Y-%m-%d")).days 
                              if isinstance(x, str) else 0)
        
        X = self.apply_to_col(X, X['sexo'], 'male', lambda x: 1 if 'H' else 0)

        X = self.apply_to_col(X, X['indresi'].fillna('S'), 'resident', lambda x: 1 if 'S' else 0)

        X = self.apply_to_col(X, X['indext'].fillna('S'), 'foreign', lambda x: 1 if 'S' else 0)

        X = self.apply_to_col(X, X['indrel'], 'indrel', lambda x: 1 if 1 else 0)

        X.indrel_1mes = X.indrel_1mes.map(lambda x: self.run_f(str(x)[0]))
        X.tiprel_1mes = X.tiprel_1mes.map(lambda x: self.run_f2(str(x)[0]))
        X.ind_empleado = X.ind_empleado.map(lambda x: 'not_emp' if x=='N' else x)
        X.ind_empleado = X.ind_empleado.map(lambda x: self.run_f3(x))
        X.segmento = X.segmento.map(lambda x: 'none_type_work' if x=='NaN' else x) 
        X.indfall = X.indfall.map(lambda x: 1 if 'S' else 0)

        cols_to_drop = ['fecha_dato','fecha_alta','ult_fec_cli_1t','cod_prov','sexo','indresi','indext','ind_nuevo']
        X = self.drop_col(X,cols_to_drop)
        return X

    def __init__(self, ide):
        self.read_path = self.get_path(ide)
        pass
        
    def __main__(self, path):
        
        # Read in the dataset
        dataset = self.read_data(path)
        # Create the feature set
        X = self.read_X(dataset, 0, 24)

        X = self.clean_dataset(X)
    
    def set_dummy_variables(self):

        d_account_type = pd.get_dummies(self.X['indrel_1mes'])
        d_employee_ind = pd.get_dummies(self.X['ind_empleado'])
        d_customer_relation = pd.get_dummies(self.X['tiprel_1mes'])
        d_channel = pd.get_dummies(self.X['canal_entrada'])
        d_province = pd.get_dummies(self.X['nomprov'])
        d_country_res = pd.get_dummies(self.X['pais_residencia'])
        d_class = pd.get_dummies(self.X['segmento'])

        # Attach all of the dummy variables to the feature set and delete the columns that
        # they came from
        self.X = pd.concat([self.X, d_employee_ind, d_account_type, d_customer_relation,
                    d_channel, d_province, d_country_res, d_class], axis=1)

        ls= ['indrel_1mes','ind_empleado', 'tiprel_1mes', 'canal_entrada', 'nomprov', 'pais_residencia', 'segmento']
        X = self.drop_col(X, ls)
        return X

        # ======================================================================================
        #                            SCALING THE CONTINUOUS VARIABLES
        # ======================================================================================
    
        
    def scaling(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()

        X.age = min_max_scaler.fit_transform(self.X.age)
        X.days_recorded_from_ny = min_max_scaler.fit_transform(self.X.days_recorded_from_ny)
        X.days_primary_churn_from_now = min_max_scaler.fit_transform(self.X.days_primary_churn_from_now)
        X.tenure_days_from_now =  min_max_scaler.fit_transform(self.X.tenure_days_from_now)
        X.antiguedad =  min_max_scaler.fit_transform(self.X.antiguedad)
        X.renta =  min_max_scaler.fit_transform(self.X.renta)

    def set_target(self, dataset):
        # Set the target set
        Y = dataset.iloc[self.X.index, 26:-1]
        Y.ind_nomina_ult1 = Y.ind_nomina_ult1.fillna(0)
        Y.ind_nom_pens_ult1 = Y.ind_nom_pens_ult1.fillna(0)
        Y.ind_nomina_ult1 = Y.ind_nomina_ult1.map(lambda x: np.int(x))
        Y.ind_nom_pens_ult1 = Y.ind_nom_pens_ult1.map(lambda x: np.int(x))

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