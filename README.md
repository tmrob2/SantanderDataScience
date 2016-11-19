# Kaggle: Santander Data Science
## Introduction

We are provided with 1.5 years of customers behavior data from Santander bank to predict what new products customers will purchase. The data starts at 2015-01-28 and has monthly records of products a customer has, such as "credit card", "savings account", etc. You will predict what additional products a customer will get in the last month, 2016-06-28, in addition to what they already have at 2016-05-28. These products are the columns named: ind_(xyz)_ult1, which are the columns #25 - #48 in the training data. You will predict what a customer will buy in addition to what they already had at 2016-05-28. 

The Santander data project is a multi label classification assignment.

In traditional supervised learning, each instance is associated with one label that indicating which class it 
belongs to. In this problem one object i.e. a customer may belong to multiple classes simultaneously. For example 
a customer may belong to a mortgage product group and a pension product group. Additionally we have a full label set 
and this is not a weak labelled problem. Therefore we are able to construct the necessary machinery using this 
information.

## A little bit of theory
To get started we are going to need a basic definition of the matrices involved. This will help with the 
formulation of the python code we will deploy to tackle this problem. Let **X** denote the feature space of the 
Santander dataset and let **L** be the set of all labels. Then for each ![xinX](https://cloud.githubusercontent.com/assets/11049017/20385856/0c420fca-acb1-11e6-9c2d-4f1f5f355224.gif) where *x* is an element of the
 feature set we have an *m* dimensional vector such that: 
 
 ![codecogseqn](https://cloud.githubusercontent.com/assets/11049017/20386050/c829622e-acb1-11e6-90ae-f17235f9a06f.gif)
 
 this represents the set set of labels for an element of **x**. Thus the set of all label values corresponding to the set **X** is the set:
 
 ![lebelSet](https://cloud.githubusercontent.com/assets/11049017/20388671/df603df8-acbd-11e6-8efd-84fc3efe46fc.gif)
 
 Similar to graph based unsupervised/ semi supervised learning we construct a positive semi definite matrix **W** where W_ij is the adjacency matrix between the i-th and j-th instances. Minimising 
 
 ![classificationBound](https://cloud.githubusercontent.com/assets/11049017/20387528/706b35e2-acb8-11e6-9427-a259baef6588.gif) 
 
 is the equivalent of the classification boundary crossing low density regions. The prediction of the k-th label F_k is the solution to the optimisation problem
 
 ![optimisationBound](https://cloud.githubusercontent.com/assets/11049017/20387550/8c0553c8-acb8-11e6-80c6-4b314926a763.gif)

where aplha and beta are controlling parameters and l(...) is the loss function. The function l(Y,f) is a convex function and therefore this is a convex optimisation problem. 

![adjmat](https://github.com/tmrob2/SantanderDataScience/blob/master/SDS/prod_adj.png)

To produce this adjacency W:

```python
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
```

## Approach

### Initial Data Size
We can calculate the size of the data input using the following, we keep the data definitions from above:
```python
    def calculate_mem_consump(self):
        nbytes = self.X.values.nbytes + self.X.index.nbytes + self.X.columns.nbytes
        return nbytes/1000000000 
```
If only a small amount of memory is available, the reasonable solution is to impot a chunksize and then calculate the memory cost of that chunk. This will give a reasonable estimate of the data granted that data is either uniformly sparse or uniformly not sparse.

Using this calcualtion the cost of the Pandas dataframe is 2.5Gb.
### Cleaning

As always the first step is to clean the data. There are a few columns that will need to be converted to dummy variables. These include:

A useful piece of code in this process was to determine the dummy variables needed to represent the data. Therefore, it is necessary to
calculate the unique elements in each feature set. One feature that was a real problem to deal with is indrel_1mes. Using:

```python
cls_data.X.indrel_1mes.value_counts(dropna=False)
```

This produces the following:

Element | Count
:---: | :---:
1.0  |  7277607
1.0  |  4017746
1    |  2195328
NaN  |   149781
3.0  |     1804
3.0  |     1409
3    |     1137
P    |      874
2    |      483
2.0  |      479
2.0  |      355
4.0  |      143
4.0  |       93
4    |       70

It is easily observed now how to deal with this data as there appears to be multiple 'unique' instances of the same values. We 
really only require one mention of 3.0 and not every variation of it.

After conducting this process for every feature the following table can be assembled as an instruction template for how we have dealt
with the feature set.

Columns Name | Data Type | Unique Values | Method
:---: | :---: | :---: | :---
fecha_dato |class 'str' | 17 | convert to days from new year and scale
ncodpers |class 'numpy.int64'| 956645 | scale
ind_empleado| class 'str'| 6 | 6 dummy variables {0,1}
pais_residencia |class 'str'| 119| 119 dummy variables {0,1}
sexo| class 'str'| 3| 1 dummy variable Male {0,1}
age |class 'str' |235 | scale
fecha_alta| class 'str' |6757| convert to days from now and scale
ind_nuevo |class 'numpy.float64'| 3 | 1 dummy variable new customer
antiguedad |class 'str' |507 | scale
indrel |class 'numpy.float64' |3 | Primary customer at month end
ult_fec_cli_1t| class 'float'| 224 | Convert to days from now and scale
indrel_1mes| class 'float' |14 | 5 dummy variables {0,1}
tiprel_1mes |class 'str' |6 | 5 dummry variables {0,1}
indresi |class 'str'| 3 | 1 dummy variable {0,1}
indext | class 'str' | 3 | 1 dummy vairable {0,1}
conyuemp | class 'float' | 3 | 1 dummy variable {0,1}
canal_entrada | class 'str'| 163 | 163 dummry variables {0,1}
indfall | class 'str' | 3 | 1 dummy variable {0,1}
tipodom | class 'numpy.float64' |2 | 1 dummy variable {0,1}
cod_prov | class 'numpy.float64'| 53 | 53 dummy variables {0,1}
nomprov | class 'str' | 53 | drop
ind_actividad_cliente | class 'numpy.float64' | 3 | 1 dummy variable {0,1}
renta | class 'numpy.float64' | 520995 | scale

This table was produced with:
```python
def return_list_of_cat_cols(self):
    for i in self.X.columns:
        print(i, type(self.X.loc[0][i]), len(self.X[i].unique()))
```
