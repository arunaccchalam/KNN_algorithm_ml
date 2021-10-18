import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV




# Data
df = pd.read_csv('../DATA/gene_expression.csv')
df.head()

# visuaizing the data
sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df,alpha=0.7)

sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df)
plt.xlim(2,6)
plt.ylim(3,10)
plt.legend(loc=(1.1,0.5))

# Train|Test Split and Scaling Data

X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scalling 
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# creating the Knn model

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(scaled_X_train,y_train)

# Understanding KNN and Choosing K Value

full_test = pd.concat([X_test,y_test],axis=1)
len(full_test)

sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=full_test,alpha=0.7)

# Model Evaluation

y_pred = knn_model.predict(scaled_X_test)

# performing metrics

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

# Elbow Method for Choosing Reasonable K Values

test_error_rates = []


for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_X_train,y_train) 
   
    y_pred_test = knn_model.predict(scaled_X_test)
    
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)

plt.figure(figsize=(10,6),dpi=200)
plt.plot(range(1,30),test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel("K Value")

# creating pipeline and and performing cross validaion for getting the best parameters

scaler = StandardScaler()
knn = KNeighborsClassifier()
knn.get_params().keys()

# Highly recommend string code matches variable name!
operations = [('scaler',scaler),('knn',knn)]

pipe = Pipeline(operations)


k_values = list(range(1,20))

k_values


param_grid = {'knn__n_neighbors': k_values}

full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')

# Use full X and y if you DON'T want a hold-out test set
# Use X_train and y_train if you DO want a holdout test set (X_test,y_test)
full_cv_classifier.fit(X_train,y_train)

full_cv_classifier.best_estimator_.get_params()

full_cv_classifier.cv_results_.keys()

len(k_values)

full_cv_classifier.cv_results_['mean_test_score']
len(full_cv_classifier.cv_results_['mean_test_score'])

# Final Model

scaler = StandardScaler()
knn14 = KNeighborsClassifier(n_neighbors=14)
operations = [('scaler',scaler),('knn14',knn14)]

pipe = Pipeline(operations)
pipe.fit(X_train,y_train)

pipe_pred = pipe.predict(X_test)

print(classification_report(y_test,pipe_pred))

single_sample = X_test.iloc[40]
single_sample
pipe.predict(single_sample.values.reshape(1, -1))
pipe.predict_proba(single_sample.values.reshape(1, -1))
