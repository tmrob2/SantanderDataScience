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
Santander dataset and let **L** be the set of all labels. Then for each ![alt text](~/PycharmProjects/SantanderDataScience/Images/xinX.gif) where *x* is an element of the
 feature set we have an *m* dimensional vector such that ![codecogseqn](https://cloud.githubusercontent.com/assets/11049017/20386050/c829622e-acb1-11e6-90ae-f17235f9a06f.gif)
