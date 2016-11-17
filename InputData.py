import numpy as np
import pandas as pd

class import_data:
    def __init__(self):
        path = '~/Data/SantanderDataScience/train_ver2.csv'

        #Get the feature set
        self.dataset = pd.read_csv(path)

        self.X = self.dataset.iloc[:, 0:23]

        #self.Y = self.dataset.iloc[:, 24:-1]



