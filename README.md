# SantanderDataScience
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
 
 this represents the set set of labels for an element of **x**. Thus the set of all labels corresponding to an instance **x** is:
 
 ![lebelSet](https://cloud.githubusercontent.com/assets/11049017/20387111/83cf57dc-acb6-11e6-8a92-65fe67fb6d08.gif)
 
 Similar to graph based unsupervised/ semi supervised learning we construct a positive semi definite matrix **W** where W_ij is the similarity matrix between the i-th and j-th instances. Minimising ![classificationBound](https://cloud.githubusercontent.com/assets/11049017/20387528/706b35e2-acb8-11e6-9427-a259baef6588.gif) is the equivalent of the classification boundary crossing low density regions. The prediction of the k-th label F_k is the solution to the optimisation problem
 
 
