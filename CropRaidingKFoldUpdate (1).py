#!/usr/bin/env python
# coding: utf-8

# In[44]:


###Crop raiding points from 2015-2021
#May 16, 2024
import numpy as np
import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


# In[68]:


#Import data
dataset=pd.read_csv("C:/Users/kem99059/Desktop/Zambezi/Incidents/MachineLearningAnalysis/HEC2015_2021_From_R_100m.csv")
print(type(dataset))
dataset.head(5)


# In[69]:


# split into input (X) and output (y) variables
X = dataset.iloc[:,1:13].values
Y = dataset.iloc[:,13].values

#Split into training and testing
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)

#Scale stuff
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)


# In[70]:


#kfold from 
#https://github.com/christianversloot/machine-learning-articles/blob/main
#/how-to-use-k-fold-cross-validation-with-keras.md

#############################
#Define keras model
#############################

# Model configuration
num_folds = 10
batch_size=200
loss_function='binary_crossentropy'
optimizer='AdaGrad'
no_epochs=5000
# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X,Y):
    
 # Define the model architecture    
    model = Sequential()
    model.add(Dense(12, input_dim=12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
        
   # Compile the model
    model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])
    
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X[train], Y[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=0)

    # Generate generalization metrics
    print(model)
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print(scores)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    #Make predictions using test data
    p_pred = model.predict(X)
    p_pred = p_pred.flatten()
    y_pred = np.where(p_pred > 0.5, 1, 0)
    #print(predictions)
    #Generate confusion matrices
    print(confusion_matrix(Y,y_pred))

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


# In[71]:


# Summarize the model
print(model.summary())


# In[37]:


## make class predictions with the model
#predictions = np.argmax(model.predict(X) > 0.5, axis=-1).astype("int32")
##predictions=(model.predict(X_test) > 0.5).astype("int32")


# In[52]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# predict probabilities for test set
yhat_probs = (model.predict(X) > 0.5).astype("int32")
# predict crisp classes for test set
yhat_classes = (model.predict(X) > 0.5).astype("int32")
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y, yhat_classes)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(Y, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(Y, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(Y, yhat_classes)
print(matrix)


# In[75]:


######
#Neural networks with Multi-layer Perceptron classifier
#######
#Import data
dataset=pd.read_csv("C:/Users/kem99059/Desktop/Zambezi/Incidents/MachineLearningAnalysis/HEC2015_2021_From_R_100m.csv")


# In[76]:


# split into input (X) and output (y) variables
X = dataset.iloc[:,1:13].values
Y = dataset.iloc[:,13].values

#Scale stuff
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)


# In[77]:


#kfold from 
#https://github.com/christianversloot/machine-learning-articles/blob/main
#/how-to-use-k-fold-cross-validation-with-keras.md
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

#############################
#Define MLP model
#############################

# Model configuration
kf = KFold(n_splits=10)
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=100000, 
                    random_state=13)

#Train the model
for train_indices,test_indices in kf.split(X):
    mlp.fit(X[train_indices], Y[train_indices.ravel()])
    #print(mlp.score(X[test_indices],Y[test_indices]))
    
    # Generate generalization metrics
    #scores = mlp.evaluate(X[test], Y[test], verbose=0)
    #print(f'Score for fold {kf}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[X]} of {scores[X]*100}%')
    #print(scores)
    #acc_per_fold.append(scores[1] * 100)
    #loss_per_fold.append(scores[0])
    #Make predictions 
    predictions = mlp.predict(X)
    print(Y)
    print(predictions)
    #Generate confusion matrices
    print(confusion_matrix(Y,predictions))
    #print(classification_report(Y,predictions))
    
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


# In[64]:


#Make predictions using test data
#predictions = mlp.predict(X)
#print(predictions)


# In[53]:


##Evaluate the model
#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(Y,predictions))
#print(classification_report(Y,predictions))


# In[66]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# predict probabilities for test set
yhat_probs = (mlp.predict(X) > 0.5).astype("int32")
# predict crisp classes for test set
yhat_classes = (mlp.predict(X) > 0.5).astype("int32")
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y, yhat_classes)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(Y, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(Y, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(Y, yhat_classes)
print(matrix)


# In[15]:


##Evaluate the model on testing data
#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))

##training data
#predict_train = mlp.predict(X_train)
#print(confusion_matrix(y_train,predict_train))
#print(classification_report(y_train,predict_train))


# In[ ]:





# In[ ]:





# In[ ]:




