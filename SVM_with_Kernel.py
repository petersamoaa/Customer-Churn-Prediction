# Hazem Samoaa - Genoa university - Data Miming project 
# Bank's customer churn modelling. 
# ML Classification Algorithm
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Data Exploration

dataset['Exited'].value_counts()
sns.countplot(x = 'Exited',data=dataset,palette='hls')
plt.show()

# explore the y as percentage to check the data balance
count_no_exit = len(dataset[dataset['Exited']==0])
count_exit = len(dataset[dataset['Exited']==1])
pct_of_no_exit = count_no_exit/(count_no_exit+count_exit)
print("percentage of no exit is", pct_of_no_exit*100)
pct_of_exit = count_exit/(count_no_exit+count_exit)
print("percentage of exit is", pct_of_exit*100)

# Our classes are imbalanced 80:20 the classes shouldn't be balanced for ANN. 

# Visualization

# Compute a simple cross-tabulation of two (or more) factors. 
# By default computes a frequency table of the factors unless an array of 
# values and an aggregation function are passed
pd.crosstab(dataset.Geography,dataset.Exited).plot(kind = "bar")
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.show()

table = pd.crosstab(dataset.Gender,dataset.Exited)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.show()

# Distribution of Age
dataset.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

del dataset,count_exit,count_no_exit,pct_of_exit,pct_of_no_exit,table                                                                     
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
# apply SMOTE 
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=[0,1,2,3,4,5,6,7,8,9,10])
os_data_y= pd.DataFrame(data=os_data_y,columns=[0])
# Show new classes balanced 
sns.countplot(x = 0,data=os_data_y,palette='hls')
plt.show()
X_train = os_data_X.copy()
y_train = os_data_y.copy()
y_train = np.ravel(y_train)
del X,y,os_data_X,os_data_y,os
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# accuracy = 84.2
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
# Accuracy mean is 0.86 & variance is 0.06
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
# 0.9039219197707736
# C =10, Gamma = 0.9 , rbf