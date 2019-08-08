# Hazem Samoaa - Genoa university - Data Miming project 
# Bank's customer churn modelling. 
# Artificial Neural Network
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
plt.title('Frequency for Geography')
plt.xlabel('Geography')
plt.ylabel('Frequency of Exit')
plt.show()

table = pd.crosstab(dataset.Gender,dataset.Exited)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Gender vs Exit')
plt.xlabel('Gender')
plt.ylabel('Proportion of Customers')
plt.show()

pd.crosstab(dataset.Tenure,dataset.Exited).plot(kind = "bar")
plt.title('Frequency for Tenure')
plt.xlabel('Tenure')
plt.ylabel('Frequency of Exit')
plt.show()

pd.crosstab(dataset.NumOfProducts,dataset.Exited).plot(kind = "bar")
plt.title('Frequency for Producst per Exit')
plt.xlabel('Products')
plt.ylabel('Frequency of Exit')
plt.show()

# Distribution of Age
dataset.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Distribution of Tenure
dataset.Tenure.hist()
plt.title('Histogram of Tenure')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.show()

# Distribution of Number of Products
dataset.NumOfProducts.hist()
plt.title('Histogram of number of product')
plt.xlabel('Number of Product')
plt.ylabel('Frequency')
plt.show()

del dataset, count_exit,count_no_exit,pct_of_exit,pct_of_no_exit,table

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
del X,y

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_