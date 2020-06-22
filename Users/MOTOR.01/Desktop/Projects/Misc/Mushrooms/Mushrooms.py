# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 08:49:16 2020

@author: MOTOR.01
"""
##########      environment setup       ##########
#import libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#import data structure
data = pd.read_csv('C:/Users/MOTOR.01/Desktop/Projects/Data/Mushrooms/mushrooms.csv')

values0 = {"e": "edible", "p":"poisonous"}
data["class"]=data["class"].replace(values0)
values={"b":"bell","c":"conical","x":"convex","f":"flat","k":"knobbed","s":"sunken"}
data["cap-shape"]=data["cap-shape"].replace(values)
values2={"f": "fibrous", "g": "grooves","y":"scaly","s": "smooth"}
data["cap-surface"]=data["cap-surface"].replace(values2)
values3={"n":"brown","b":"buff","c":"cinnamon","g":"gray","r":"green","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}
data["cap-color"]=data["cap-color"].replace(values3)
values4={"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"}
data["odor"]=data["odor"].replace(values4)
values5={"a":"attached","f":"free"}
data["gill-attachment"]=data["gill-attachment"].replace(values5)
values6={"c":"close","w":"crowded"}
data["gill-spacing"]=data["gill-spacing"].replace(values6)
values7={"b":"broad","n":"narrow"}
data["gill-size"]=data["gill-size"].replace(values7)
values8={"k":"black","b":"buff","n":"brown","h":"chocolate","g":"gray","r":"green","o":"orange","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}
data["gill-color"]=data["gill-color"].replace(values8)
values9={"t":"tapering","e":"enlarging"}
data["stalk-shape"]=data["stalk-shape"].replace(values9)
values10={"b":"bulbous","c":"club","e":"equal","z":"rhizomorphs","r":"rooted","?":"missing"}
data["stalk-root"]=data["stalk-root"].replace(values10)
values11={"s":"smooth","k":"silky","f":"fibrous","y":"scaly"}
data["stalk-surface-above-ring"]=data["stalk-surface-above-ring"].replace(values11)
data["stalk-surface-below-ring"]=data["stalk-surface-below-ring"].replace(values11)
values12={"n":"brown","b":"buff","c":"cinnamon","g":"gray","p":"pink","e":"red","w":"white","y":"yellow","o":"orange"}
data["stalk-color-above-ring"]=data["stalk-color-above-ring"].replace(values12)
data["stalk-color-below-ring"]=data["stalk-color-below-ring"].replace(values12)
veil_type={"p":"partial","u":"universal"} 
data["veil-type"]=data["veil-type"].replace(veil_type)
veil_color={"n":"brown","o":"orange","w":"white","y":"yellow"} 
data["veil-color"]=data["veil-color"].replace(veil_color)
ring_number= {"n":"none","o":"one","t":"two"}
data["ring-number"]=data["ring-number"].replace(ring_number)
ring_type={"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"}
data["ring-type"]=data["ring-type"].replace(ring_type)
spore_print_color= {"k":"black","n":"brown","b":"buff","h":"chocolate","r":"green","o":"orange","u":"purple","w":"white","y":"yellow"}
data["spore-print-color"]=data["spore-print-color"].replace(spore_print_color)
population={"a":"abundant","c":"clustered","n":"numerous","s":"scattered","v":"several","y":"solitary"}
data["population"]=data["population"].replace(population)
habitat={"g":"grasses","l":"leaves","m":"meadows","p":"paths","u":"urban","w":"waste","d":"woods"}
data["habitat"]=data["habitat"].replace(habitat)
bruises={"t":"bruises","f":"no"}
data["bruises"]=data["bruises"].replace(bruises)

##########      exploratory data analysis       ##########
### data structure
data.columns
        #'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        #'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        #'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        #'stalk-surface-below-ring', 'stalk-color-above-ring',
        #'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        #'ring-type', 'spore-print-color', 'population', 'habitat'

data.head()
data.shape
data.info()
data.describe()
### check for nulls
null_sum = data.isnull().sum()
null_sum = null_sum[null_sum > 0]
null_proportion = null_sum * 100 / data.shape[0]
pd.concat([null_sum, null_proportion], axis = 1, 
          keys = ['Missing Values', 'Percentage']).sort_values(by = "Missing Values", ascending = False)

### variable visualization
color_scheme = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive")

#subplots - cap
fig, axarr = plt.subplots(2, 2, figsize = (20,20))

data['cap-shape'].value_counts().plot.bar(
    ax = axarr[0][0], fontsize = 12, color = color_scheme, edgecolor = "black")
axarr[0][0].set_title("Cap Shape", fontsize = 18)

data['cap-surface'].value_counts().plot.bar(
    ax = axarr[1][0], fontsize = 12, color = color_scheme, edgecolor = "black")
axarr[1][0].set_title("Cap Surface", fontsize = 18)

data['cap-color'].value_counts().plot.bar(
    ax = axarr[1][1], fontsize = 12, 
    color = ("tab:brown", "gray", "tab:red", "yellow", "whitesmoke", "goldenrod", "pink", "chocolate", "purple", "green"), 
    edgecolor = "black")
axarr[1][1].set_title("Cap Color", fontsize = 18)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 45)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

#subplot - gill
fig, axarr = plt.subplots(2, 2, figsize = (20,20))

data['gill-attachment'].value_counts().plot.bar(
    ax = axarr[0][0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[0][0].set_title("Gill Attachment", fontsize = 20)

data['gill-spacing'].value_counts().plot.bar(
    ax = axarr[1][0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[1][0].set_title("Gill Spacing", fontsize = 20)

data['gill-size'].value_counts().plot.bar(
    ax = axarr[1][1], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[1][1].set_title("Gill Size", fontsize = 20)

data['gill-color'].value_counts().plot.bar(
    ax = axarr[0][1], fontsize = 12, color = color_scheme, edgecolor = "black")
axarr[0][1].set_title("Gill Color", fontsize = 20)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 0)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

#subplot - stalk
fig, axarr = plt.subplots(3, 2, figsize = (20, 25))

data['stalk-shape'].value_counts().plot.bar(
    ax = axarr[0][0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[0][0].set_title("Stalk Shape", fontsize = 20)

data['stalk-root'].value_counts().plot.bar(
    ax = axarr[1][0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[1][0].set_title("Stalk Root", fontsize = 20)

data['stalk-surface-above-ring'].value_counts().plot.bar(
    ax = axarr[1][1], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[1][1].set_title("Stalk Surface Above Ring", fontsize = 20)

data['stalk-surface-below-ring'].value_counts().plot.bar(
    ax = axarr[0][1], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[0][1].set_title("Stalk Surface Below Ring", fontsize = 20)

data['stalk-color-above-ring'].value_counts().plot.bar(
    ax = axarr[2][0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[2][0].set_title("Stalk Color Above Ring", fontsize = 20)

data['stalk-color-below-ring'].value_counts().plot.bar(
    ax = axarr[2][1], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[2][1].set_title("Stalk Color Below Ring", fontsize = 20)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 0)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    
#subplot - veil
fig, axarr = plt.subplots(1, 2, figsize = (10, 5))

data['veil-type'].value_counts().plot.bar(
    ax = axarr[0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[0].set_title("Veil Type", fontsize = 20)

data['veil-color'].value_counts().plot.bar(
    ax = axarr[1], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[1].set_title("Veil Color", fontsize = 20)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 0)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

#subplot - ring
fig, axarr = plt.subplots(1, 2, figsize = (10, 5))

data['ring-number'].value_counts().plot.bar(
    ax = axarr[0], fontsize = 12, color = color_scheme, edgecolor = "black")
axarr[0].set_title("Ring Number", fontsize = 20)

data['ring-type'].value_counts().plot.bar(
    ax = axarr[1], fontsize = 12, color = color_scheme, edgecolor = "black")
axarr[1].set_title("Ring Type", fontsize = 20)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 45)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    
#subplot - other
fig, axarr = plt.subplots(3, 2, figsize = (20, 25))

data['bruises'].value_counts().plot.bar(
    ax = axarr[0][0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[0][0].set_title("Bruises", fontsize = 20)

data['odor'].value_counts().plot.bar(
    ax = axarr[1][0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[1][0].set_title("Odor", fontsize = 20)

data['spore-print-color'].value_counts().plot.bar(
    ax = axarr[1][1], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[1][1].set_title("Spore Print color", fontsize = 20)

data['population'].value_counts().plot.bar(
    ax = axarr[0][1], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[0][1].set_title("Population", fontsize = 20)

data['habitat'].value_counts().plot.bar(
    ax = axarr[2][0], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[2][0].set_title("Habitat", fontsize = 20)

data['class'].value_counts().plot.bar(
    ax = axarr[2][1], fontsize = 15, color = color_scheme, edgecolor = "black")
axarr[2][1].set_title("Class", fontsize = 20)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 0)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

##########      modeling       ##########
#label encoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

data = data.apply(label_encoder.fit_transform)

data.head()
data.info()

y = data["class"].values
data.drop(["class"], axis = 1, inplace = True)

x_data = data

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x.drop(["veil-color"], axis = 1, inplace = True)
x.drop(["veil-type"], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#transpose matrices
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

#logistic regression
def init_weight_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b;

def sigmoid(z):
    y_head = 1/(1+ np.exp(-z))
    return y_head;

def forward_and_backward_propagation(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss)) / x_train.shape[1]
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/ x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"deriv_weight":derivative_weight,"deriv_bias":derivative_bias}
    return cost, gradients;

def update(w,b,x_train,y_train,learning_rate,num_of_iter):
    cost_list = []   # empty arrays to store costs
    index = []
    
    for i in range(num_of_iter):
        cost, gradients = forward_and_backward_propagation(w,b,x_train,y_train)  # do the training as much as iteration given
      #  print("update:: ",cost)
        cost_list.append(cost)      # insert the calculated cost on array
        index.append(i)
        
        w = w - learning_rate*gradients["deriv_weight"]     # set new weights and bias for next iteration
        b = b - learning_rate*gradients["deriv_bias"]
        
    parameters = {"weight":w,"bias":b}    # save all the weights and biases on a dictionary
    plt.plot(index,cost_list)            # draw the plot to visualize (optional)
    plt.show()
    
    return parameters, gradients, cost_list;


def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test) + b)           # do the first and second phase of Computation and store in array z
    y_prediction = np.zeros((1,x_test.shape[1]))  # create empty array to fill by results of z
   # print("y_pred:::", np.zeros((1,x_test.shape[1])))
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0;
        else:
            y_prediction[0,i] = 1;
    
    return y_prediction; 

def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_of_iter):
    
    dimension = x_train.shape[0]
    w,b = init_weight_and_bias(dimension)
    
    parameters, gradients, col_list = update(w,b,x_train,y_train,learning_rate,num_of_iter)
    
    y_pred_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_pred_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    print("train accuracy: {} %".format(100-np.mean(np.abs(y_pred_train-y_train))*100))
    print("test accuracy: {} %".format(100-np.mean(np.abs(y_pred_test-y_test))*100))
 

logistic_regression(x_train,y_train,x_test,y_test,learning_rate=1,num_of_iter=250)

#sklearn version
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(x_train.T,y_train.T)
y_head = lr_model.predict(x_test.T)
print("test accuracy: ", lr_model.score(x_test.T,y_test.T))


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_head)

plt.figure(figsize=(16,10))
sns.heatmap(cm,annot=True,fmt='.0f')
plt.show()

#coefficients
coeff_data = pd.DataFrame(lr_model.coef_, x.columns, columns = ['Coefficient'])
coeff_data

#plots
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_head})
df1 = df.head(25)

df1.plot(kind = 'bar', figsize = (10,8))
plt.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'gray')
plt.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black')
plt.show()

#measures
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_head))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_head))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_head)))
