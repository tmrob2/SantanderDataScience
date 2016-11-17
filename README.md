# Kaggle: Santander Data Science
## Introduction
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

As always the first step is to clean the data. There are a few columns that will need to be converted to dummy variables. These include:

1. ind_empleado
2. sexo
3. indresi
4. indext
5. conuemp
6. canal_entrada
7. indfall
8. nomprov

