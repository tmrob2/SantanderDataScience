import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
import networkx as nx 
import matplotlib.pyplot as plt
import itertools
import math

class ImportData:

    read_path = {}
    feature_list = []

    def get_path(self, ide, filename):
        if ide == 'vs':
            path = 'C:/Users/Thomas/Documents/Data/Santander/%s'%filename
        elif ide == 'spyder':
            path = 'C:/Users/611004435/Documents/Data/Santander/%s'%filename
        else:
            path = '~/Data/SantanderDataScience/%s'%filename
        return path

    def read_random_data(self, path, sample_size, chunk_size):
        """
        choose a sample size and randomly select the rows to skip effectively having a sample that 
        randomly represents the population.
        """
        dataset = pd.DataFrame()
        nlinesfile= 14000000
        nlines_rnd_sampl = sample_size
        lines2skip = np.random.choice(np.arange(1,nlinesfile+1), 
                                        (nlinesfile-nlines_rnd_sampl), replace=False)
        for chunk in pd.read_csv(path, skiprows=lines2skip, chunksize=chunk_size, low_memory=False):
            dataset = dataset.append(chunk)
        return dataset

    def read_X(self, dataset: pd.DataFrame, col_from:int, col_to:int):
        """ Allocate the feture set from the dataset"""
        X = dataset.iloc[:, col_from:col_to]
        return X

    def train_cols(self,X):
        cols = X.columns.values
        return cols

    def compare_cols_and_insert(self, cols_train, X_test):
        """
        Assumes that the column missing is categorical
        """
        for i in cols_train:
            if i not in X_test.columns.values:
                X_test[i] = [0]

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
        i. Remove the rows without a tenure date
        ii. Replace nans with appropriate values
        iii. Convert dates
        iv. Replace {a,b} -> {0,1}
        v. Apply more robust descriptive categorical variables to apprpriate columns: these will 
        vi. be converted to dummy variables
        """

        X = self.remove_nan_rows(X,'fecha_alta', ['NaN','NA',' NA']) 
        X = self.remove_nan_rows(X, 'age', ['NaN','NA',' NA'])
        X['new_cust'] = X['ind_nuevo'].fillna(0)
        X.conyuemp = self.apply_to_col(X, X.conyuemp.fillna(0), 'spouse_account',  lambda x: 1 if 'S' else 0) 
        
        X = self.apply_to_col(X, X['age'].fillna(0), 'age', lambda x: np.float(x))
        X = self.apply_to_col(X, X['fecha_dato'], 'days_recorded_from_ny',
                                     lambda x: (datetime.datetime.strptime(x,"%Y-%m-%d" )-
                                     datetime.datetime(2015,1,1)).days)

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

        X = self.apply_to_col(X, X['renta'].fillna(0), 'earnings', lambda x: np.float(str(x).strip()) if str(x).strip() != 'NA' else 0.0)

        X.indrel_1mes = X.indrel_1mes.map(lambda x: self.run_f(str(x)[0]))
        X.tiprel_1mes = X.tiprel_1mes.map(lambda x: self.run_f2(str(x)[0]))
        X.ind_empleado = X.ind_empleado.map(lambda x: 'not_emp' if x=='N' else x)
        X.ind_empleado = X.ind_empleado.map(lambda x: self.run_f3(x))
        X.segmento = X.segmento.map(lambda x: 'none_type_work' if x=='NaN' else x) 
        X.indfall = X.indfall.map(lambda x: 1 if 'S' else 0)

        cols_to_drop = ['fecha_dato','fecha_alta','ult_fec_cli_1t',
                        'cod_prov','sexo','indresi','indext','ind_nuevo', 'renta', 'conyuemp']
        X = self.drop_col(X,cols_to_drop)
        return X

    def __init__(self, ide, filename):
        self.read_path[filename] = self.get_path(ide, filename)
        pass

    def read_another_file(self, ide, filename):
        self.read_path[filename] = self.get_path(ide, filename)
        
    def __main__(self, path, import_type:str='train', sample_size = 50000):
        
        # Read in the dataset
        if import_type == 'random':
            dataset = self.read_random_data(path, sample_size, 1000000)
        elif import_type == 'test':
            dataset = pd.read_csv(path, low_memory=False)
        else:
            dataset = pd.read_csv(path, low_memory=False)

        # Create the feature set
        X = self.read_X(dataset, 0, 24)

        X = self.clean_dataset(X)

        X = self.set_dummy_variables(X)

        #X = self.scaling(X)

        Y = self.set_target(dataset, X)

        if import_type != 'test':
            Y = self.clean_target_set(Y) 
        return X,Y 
    
    def set_dummy_variables(self, X):

        d_account_type = pd.get_dummies(X['indrel_1mes'])
        d_employee_ind = pd.get_dummies(X['ind_empleado'])
        d_customer_relation = pd.get_dummies(X['tiprel_1mes'])
        d_channel = pd.get_dummies(X['canal_entrada'])
        d_province = pd.get_dummies(X['nomprov'])
        d_country_res = pd.get_dummies(X['pais_residencia'])
        d_class = pd.get_dummies(X['segmento'])

        # Attach all of the dummy variables to the feature set and delete the columns that
        # they came from
        X = pd.concat([X, d_employee_ind, d_account_type, d_customer_relation,
                    d_channel, d_province, d_country_res, d_class], axis=1)

        ls= ['indrel_1mes','ind_empleado', 
             'tiprel_1mes', 'canal_entrada', 'nomprov', 'pais_residencia', 'segmento']
        X = self.drop_col(X, ls)
        return X
        
    def scaling(self, X):
        #min_max_scaler = preprocessing.MinMaxScaler()
        #X.age = min_max_scaler.fit_transform(X.age)
        ls = ['ncodpers','age','antiguedad','days_recorded_from_ny','tenure','days_churned','earnings']

        for i in ls:
            X[i] = preprocessing.scale(X[i]) 
        return X

    def set_target(self, dataset, X):
        # Set the target set
        Y = dataset.iloc[X.index, 26:-1]
        return Y

    def clean_target_set(self, Y):
        Y.ind_nomina_ult1 = Y.ind_nomina_ult1.fillna(0)
        Y.ind_nom_pens_ult1 = Y.ind_nom_pens_ult1.fillna(0)
        Y.ind_nomina_ult1 = Y.ind_nomina_ult1.map(lambda x: np.int(x))
        Y.ind_nom_pens_ult1 = Y.ind_nom_pens_ult1.map(lambda x: np.int(x))
        return Y

    def return_list_of_cat_cols(self, X):
        for i in X.columns:
            print(i, type(X.loc[0][i]), len(X[i].unique()))

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