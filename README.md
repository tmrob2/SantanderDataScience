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
 
 Similar to graph based unsupervised/ semi supervised learning we construct a positive semi definite matrix **W** where W_ij is the similarity matrix between the i-th and j-th instances. Minimising 
 
 ![classificationBound](https://cloud.githubusercontent.com/assets/11049017/20387528/706b35e2-acb8-11e6-9427-a259baef6588.gif) 
 
 is the equivalent of the classification boundary crossing low density regions. The prediction of the k-th label F_k is the solution to the optimisation problem
 
 ![optimisationBound](https://cloud.githubusercontent.com/assets/11049017/20387550/8c0553c8-acb8-11e6-80c6-4b314926a763.gif)

where aplha and beta are controlling parameters and l(...) is the loss function. 

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
Columns Name | Data Type | Unique Values
--- | --- | ---
fecha_dato |<class 'str'> | 17
ncodpers |<class 'numpy.int64'>| 956645
ind_empleado| <class 'str'>| 6
pais_residencia |<class 'str'>| 119
sexo| <class 'str'>| 3
age |<class 'str'> |235
fecha_alta| <class 'str'> |6757
ind_nuevo |<class 'numpy.float64'>| 3
antiguedad |<class 'str'> |507
indrel |<class 'numpy.float64'> |3
ult_fec_cli_1t| <class 'float'>| 224
indrel_1mes| <class 'float'> |14
tiprel_1mes |<class 'str'> |6
indresi |<class 'str'>| 3
indext | <class 'str'> | 3
conyuemp | <class 'float'> | 3
canal_entrada | <class 'str'>| 163
indfall | <class 'str'> | 3
tipodom | <class 'numpy.float64'> |2
cod_prov | <class 'numpy.float64'>| 53
nomprov | <class 'str'> | 53
ind_actividad_cliente | <class 'numpy.float64'> | 3
renta | <class 'numpy.float64'>| 520995

