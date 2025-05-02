Decision Trees and Random Forests


 Objective: Learn tree-based models for classification & regression. 

 
 Tools:  Scikit-learn
 


 features
 1.Train a Decision Tree Classifier and visualize the tree.
 2.Analyze overfitting and control tree depth.
 3.Train a Random Forest and compare accuracy.
 4.Interpret feature importances.
 5.Evaluate using cross-validation




 | Model                         | Test Accuracy | Cross-Validation Accuracy |
| ----------------------------- | ------------- | ------------------------- |
| **Decision Tree** (depth=5)   | 0.844         | 0.891                     |
| **Random Forest** (100 trees) | 0.985         | 0.997                     |



🔍 Interpretation
Random Forest outperforms the Decision Tree significantly, both on the test set and under cross-validation.

It generalizes better by reducing overfitting through ensemble learning.

The most important features (based on feature importance plot) likely include variables such as:

cp (chest pain type)

thalach (max heart rate achieved)

ca (number of major vessels)

oldpeak (ST depression)



## code



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df=pd.read_csv("heart (1).csv")
df.head()
age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
0	52	1	0	125	212	0	1	168	0	1.0	2	2	3	0
1	53	1	0	140	203	1	0	155	1	3.1	0	0	3	0
2	70	1	0	145	174	0	1	125	1	2.6	0	0	3	0
3	61	1	0	148	203	0	1	161	0	0.0	2	1	3	0
4	62	0	0	138	294	1	1	106	0	1.9	1	3	2	0
df.tail()
age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
1020	59	1	1	140	221	0	1	164	1	0.0	2	0	2	1
1021	60	1	0	125	258	0	0	141	1	2.8	1	1	3	0
1022	47	1	0	110	275	0	0	118	1	1.0	1	1	2	0
1023	50	0	0	110	254	0	0	159	0	0.0	2	0	2	1
1024	54	1	0	120	188	0	1	113	0	1.4	1	1	3	0
df.describe()
age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
count	1025.000000	1025.000000	1025.000000	1025.000000	1025.00000	1025.000000	1025.000000	1025.000000	1025.000000	1025.000000	1025.000000	1025.000000	1025.000000	1025.000000
mean	54.434146	0.695610	0.942439	131.611707	246.00000	0.149268	0.529756	149.114146	0.336585	1.071512	1.385366	0.754146	2.323902	0.513171
std	9.072290	0.460373	1.029641	17.516718	51.59251	0.356527	0.527878	23.005724	0.472772	1.175053	0.617755	1.030798	0.620660	0.500070
min	29.000000	0.000000	0.000000	94.000000	126.00000	0.000000	0.000000	71.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	48.000000	0.000000	0.000000	120.000000	211.00000	0.000000	0.000000	132.000000	0.000000	0.000000	1.000000	0.000000	2.000000	0.000000
50%	56.000000	1.000000	1.000000	130.000000	240.00000	0.000000	1.000000	152.000000	0.000000	0.800000	1.000000	0.000000	2.000000	1.000000
75%	61.000000	1.000000	2.000000	140.000000	275.00000	0.000000	1.000000	166.000000	1.000000	1.800000	2.000000	1.000000	3.000000	1.000000
max	77.000000	1.000000	3.000000	200.000000	564.00000	1.000000	2.000000	202.000000	1.000000	6.200000	2.000000	4.000000	3.000000	1.000000
df.size
14350
df.shape
(1025, 14)
df.isnull().sum()
age         0
sex         0
cp          0
trestbps    0
chol        0
fbs         0
restecg     0
thalach     0
exang       0
oldpeak     0
slope       0
ca          0
thal        0
target      0
dtype: int64
df.duplicated().sum()
723
df1=df.drop_duplicates()
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1025 entries, 0 to 1024
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1025 non-null   int64  
 1   sex       1025 non-null   int64  
 2   cp        1025 non-null   int64  
 3   trestbps  1025 non-null   int64  
 4   chol      1025 non-null   int64  
 5   fbs       1025 non-null   int64  
 6   restecg   1025 non-null   int64  
 7   thalach   1025 non-null   int64  
 8   exang     1025 non-null   int64  
 9   oldpeak   1025 non-null   float64
 10  slope     1025 non-null   int64  
 11  ca        1025 non-null   int64  
 12  thal      1025 non-null   int64  
 13  target    1025 non-null   int64  
dtypes: float64(1), int64(13)
memory usage: 112.2 KB
df1.info()
<class 'pandas.core.frame.DataFrame'>
Index: 302 entries, 0 to 878
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       302 non-null    int64  
 1   sex       302 non-null    int64  
 2   cp        302 non-null    int64  
 3   trestbps  302 non-null    int64  
 4   chol      302 non-null    int64  
 5   fbs       302 non-null    int64  
 6   restecg   302 non-null    int64  
 7   thalach   302 non-null    int64  
 8   exang     302 non-null    int64  
 9   oldpeak   302 non-null    float64
 10  slope     302 non-null    int64  
 11  ca        302 non-null    int64  
 12  thal      302 non-null    int64  
 13  target    302 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 35.4 KB
from sklearn.tree import DecisionTreeClassifier  # for classification
from sklearn.tree import DecisionTreeRegressor   # for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
x = df.drop("target", axis=1)
y = df["target"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
dt= DecisionTreeClassifier()
dt.fit(xtrain, ytrain)

  DecisionTreeClassifier?i
DecisionTreeClassifier()
ypred = dt.predict(xtest)
print("Accuracy:", accuracy_score(ytest, ypred))
Accuracy: 0.9853658536585366
depths = range(1, 21)
train_acc = []
test_acc = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(xtrain, ytrain)
    train_acc.append(tree.score(xtrain, ytrain))
    test_acc.append(tree.score(xtest, ytest))

# Plot accuracy vs tree depth
plt.figure(figsize=(10, 6))
plt.plot(depths, train_acc, label="Train Accuracy")
plt.plot(depths, test_acc, label="Test Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting Analysis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
No description has been provided for this image
from sklearn.model_selection import cross_val_score
import seaborn as sns
optimal_depth = 5
dt = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
dt.fit(xtrain, ytrain)
dt_pred = dt.predict(xtest)
dt_acc = accuracy_score(ytest, dt_pred)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(xtrain, ytrain)
rf_pred = rf.predict(xtest)
rf_acc = accuracy_score(ytest, rf_pred)
dt_cv_score = cross_val_score(dt, x, y, cv=5).mean()
rf_cv_score = cross_val_score(rf, x, y, cv=5).mean()
# 4. Feature importance from Random Forest
importances = rf.feature_importances_
feature_names = x.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
No description has been provided for this image
 # Results 
{"Decision Tree Test Accuracy": round(dt_acc, 3),
"Random Forest Test Accuracy": round(rf_acc, 3),
"Decision Tree CV Accuracy": round(dt_cv_score, 3),
"Random Forest CV Accuracy": round(rf_cv_score, 3)}
{'Decision Tree Test Accuracy': 0.844,
 'Random Forest Test Accuracy': 0.985,
 'Decision Tree CV Accuracy': 0.891,
 'Random Forest CV Accuracy': 0.997}
 
 
