Ex.No. 1	Implementation of Decision Tree Algorithm	Date:

Aim:

Write a program to demonstrate the working of the decision tree algorithm.
Program:
	#Three lines to make our compiler able to draw:
import sys
import matplotlib
%matplotlib inline
#matplotlib.use('Agg')

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
df = pandas.read_csv("D:\ML\data.csv")
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
fig=plt.figure(figsize=(15,20))
fig=tree.plot_tree(dtree, feature_names=features)

#Two  lines to make our compiler able to draw:
plt.savefig("dtree.png")
#plt.savefig(sys.stdout.buffer)
#fig=plt.figure(figsize=(15,20))
sys.stdout.flush()
print(df)
print(dtree.predict([[35,10,1,1]]))



Dataset:
	
Age	Experience	Rank	Nationality	Go
36	10	9	UK	NO
42	12	4	USA	NO
23	4	6	N	NO
52	4	4	USA	NO
43	21	8	USA	YES
44	14	5	UK	NO
66	3	7	N	YES
35	14	9	UK	YES
52	13	7	N	YES
35	5	9	N	YES
24	3	5	USA	NO
18	3	7	UK	YES
45	9	9	UK	YES

--------------------------------------------------------------------------------------------------------------------------------------------------------------


Ex.No. 2	Back Propagation Algorithm.	Date:

Aim:
Write a program for implementing the Back propagation algorithm and test the same using appropriate datasets.
	
Program:
import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0)
# maximum of X array longitudinally 
y = y/100
#Sigmoid Function 
def sigmoid (x): return 1/(1 + np.exp(-x))
#Derivative of Sigmoid Function
def derivatives_sigmoid(x): return x * (1 - x)
#Variable initialization
epoch=5000	
#Setting training iterations
lr=0.1
#Setting learning rate
inputlayer_neurons = 2	
#number of features in data set 
hiddenlayer_neurons = 3	
#number of hidden layers neurons 
output_neurons = 1	
#number of neurons at output layer 
#weight and bias initialization 
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons)) 
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
#draws a random range of numbers uniformly of dim x*y  for i in range(epoch):
#Forward Propogation
hinp1=np.dot(X,wh) 
hinp=hinp1 + bh
hlayer_act = sigmoid(hinp) 
outinp1=np.dot(hlayer_act,wout) 
outinp= outinp1+ bout
output = sigmoid(outinp)
#Backpropagation 
EO = y-output
outgrad = derivatives_sigmoid(output) 
d_output = EO* outgrad
EH = d_output.dot(wout.T)
#how much hidden layer wts contributed to error 
hiddengrad = derivatives_sigmoid(hlayer_act) 
d_hiddenlayer = EH * hiddengrad
# dotproduct of nextlayererror and currentlayerop
wout += hlayer_act.T.dot(d_output) *lr
wh+= X.T.dot(d_hiddenlayer) *lr
print("Input: \n" + str(X)) 
print("Actual Output: \n" + str(y)) 
print("Predicted Output: \n" ,output)

Output:
Input: 
[[0.66666667 1.        ]
 [0.33333333 0.55555556]
 [1.         0.66666667]]

Actual Output: 
[[0.92]	
 [0.86]
 [0.89]]

Predicted Output: 
 [[0.92745804]
 [0.91954311]
 [0.92598481]]
--------------------------------------------------------------------------------------------------------------
Ex.No. 4	Naïve Bayesian Classifier	Date:

Aim:
To write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.

Program:

import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('tennisdata.csv')
print("The first 5 values of data is :\n",data.head())

X = data.iloc[:,:-1]
print("\nThe First 5 values of train data is\n",X.head())
y = data.iloc[:,-1]
print("\nThe first 5 values of Train output is\n",y.head())

le_outlook = LabelEncoder()
X.Outlook = le_outlook.fit_transform(X.Outlook)
le_Temperature = LabelEncoder()
X.Temperature = le_Temperature.fit_transform(X.Temperature)
le_Humidity = LabelEncoder()
X.Humidity = le_Humidity.fit_transform(X.Humidity)
le_Windy = LabelEncoder()
X.Windy = le_Windy.fit_transform(X.Windy)

print("\nNow the Train data is :\n",X.head())
le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is\n",y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
classifier = GaussianNB()
classifier.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print("Accuracy is:",accuracy_score(classifier.predict(X_test),y_test))





Tennisdata.csv

Outlook	Temperature	Humidity	Windy	PlayTennis
Sunny	Hot	High	FALSE	No
Sunny	Hot	High	TRUE	No
Overcast	Hot	High	FALSE	Yes
Rainy	Mild	High	FALSE	Yes
Rainy	Cool	Normal	FALSE	Yes
Rainy	Cool	Normal	TRUE	No
Overcast	Cool	Normal	TRUE	Yes
Sunny	Mild	High	FALSE	No
Sunny	Cool	Normal	FALSE	Yes
Rainy	Mild	Normal	FALSE	Yes
Sunny	Mild	Normal	TRUE	Yes
Overcast	Mild	High	TRUE	Yes
Overcast	Hot	Normal	FALSE	Yes
Rainy	Mild	High	TRUE	No

Output:


The first 5 values of data is :
     Outlook Temperature Humidity  Windy PlayTennis
0     Sunny         Hot            High       False         No
1     Sunny         Hot            High       True         No
2  Overcast         Hot           High       False        Yes
3     Rainy        Mild            High       False        Yes
4     Rainy        Cool            Normal   False        Yes

The First 5 values of train data is
     Outlook Temperature Humidity  Windy
0     Sunny         Hot           High         False
1     Sunny         Hot           High         True
2   Overcast       Hot          High          False
3     Rainy        Mild           High          False
4     Rainy        Cool           Normal      False



The first 5 values of Train output is
0     No
1     No
2    Yes
3    Yes
4    Yes
Name: PlayTennis, dtype: object

Now the Train data is :
Outlook     Temperature    Humidity    Windy
0              2               1                       0              0
1       	      2               1                       0              1
2              0               1                       0              0
3              1               2                       0              0
4              1               0                       1              0

Now the Train output is
 [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
Accuracy is: 0.3333333333333333

----------------------------------------------------------------------------------------------------------------
Ex.No.6	Bayesian network considering Medical Dataset	Date:

Aim:
Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease DataSet. You can use Java/Python ML library classes/API.
Program:

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
data = pd.read_csv(r"C:\Users\Sirisha\Downloads\ds4.csv")
heart_disease = pd.DataFrame(data)
print(heart_disease)
model = BayesianModel([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease'),
    ('diet', 'cholestrol')
])
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)
HeartDisease_infer = VariableElimination(model)
print('For Age enter SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4')
print('For Gender enter Male:0, Female:1')
print('For Family History enter Yes:1, No:0')
print('For Diet enter High:0, Medium:1')
print('for LifeStyle enter Athlete:0, Active:1, Moderate:2, Sedentary:3')
print('for Cholesterol enter High:0, BorderLine:1, Normal:2')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
    'age': int(input('Enter Age: ')),
    'Gender': int(input('Enter Gender: ')),
    'Family': int(input('Enter Family History: ')),
    'diet': int(input('Enter Diet: ')),
    'Lifestyle': int(input('Enter Lifestyle: ')),
    'cholestrol': int(input('Enter Cholestrol: '))
})

print(q)

Dataset:

age	Gender	Family	diet	Lifestyle	cholesterol	heartdisease
0	0	1	1	3	0	1
0	1	1	1	3	0	1
1	0	0	0	2	1	1
4	0	1	1	3	2	0
3	1	1	0	0	2	0
2	0	1	1	1	0	1
4	0	1	0	2	0	1
0	0	1	1	3	0	1
3	1	1	0	0	2	0
1	1	0	0	0	2	1
4	1	0	1	2	0	1
4	0	1	1	3	2	0
2	1	0	0	0	0	0
2	0	1	1	1	0	1
3	1	1	0	0	1	0
0	0	1	0	0	2	1
1	1	0	1	2	1	1
3	1	1	1	0	1	0
4	0	1	1	3	2	0

Output:
For Age enter SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4
For Gender enter Male:0, Female:1
For Family History enter Yes:1, No:0
For Diet enter High:0, Medium:1
for LifeStyle enter Athlete:0, Active:1, Moderate:2, Sedentary:3
for Cholesterol enter High:0, BorderLine:1, Normal:2
Enter Age: 2
Enter Gender: 1
Enter Family History: 0
Enter Diet: 1
Enter Lifestyle: 2
Enter Cholestrol: 1

+-----------------+---------------------+
| heartdisease    |   phi(heartdisease) |
+=================+=====================+
| heartdisease(0) |              0.5000 |
+-----------------+---------------------+
| heartdisease(1) |              0.5000 |
+-----------------+---------------------+
------------------------------------------------------------------------------------
Ex.No. 7	Expectation Maximization Algorithm	Date:

Aim:
Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same Dataset for clustering using k-Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add Java/Python ML library classes / API in the program.
Program:

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=load_iris()
# print(dataset)
X=pd.DataFrame(dataset.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y=pd.DataFrame(dataset.target)
y.columns=['Targets']
# print(X)
plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])
# REAL PLOT
plt.subplot(1,3,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y.Targets],s=40)
plt.title('Real')
# K-PLOT
plt.subplot(1,3,2)
model=KMeans(n_clusters=3)
model.fit(X)


predY=np.choose(model.labels_,[0,1,2]).astype(np.int64)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[predY],s=40)
plt.title('KMeans')

Output:

 

--------------------------------------------------------------------------------
Ex.No. 8	Principle Component Analysis for Dimensionality Reduction.	Date:

Aim:
Write a program to implement Principle Component Analysis for Dimensionality Reduction.
Program:
	
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# define a small 3×2 matrix
matrix = array([[5, 6], [8, 10], [12, 18]])
print("original Matrix: ")
print(matrix)

# calculate the mean of each column
Mean_col = mean(matrix.T, axis=1)
print("Mean of each column: ")
print(Mean_col)

# center columns by subtracting column means
Centre_col = matrix - Mean_col
print("Covariance Matrix: ")
print(Centre_col)

# calculate covariance matrix of centered matrix
cov_matrix = cov(Centre_col.T)
print(cov_matrix)

# eigendecomposition of covariance matrix
values, vectors = eig(cov_matrix)
print("Eigen vectors: ",vectors)
print("Eigen values: ",values)

# project data on the new axes
projected_data = vectors.T.dot(Centre_col.T)
print(projected_data.T)
	 


	Output:

original Matrix: 
[[ 5  6]
 [ 8 10]
 [12 18]]
Mean of each column: 
[ 8.33333333 11.33333333]
Covariance Matrix: 
[[-3.33333333 -5.33333333]
 [-0.33333333 -1.33333333]
 [ 3.66666667  6.66666667]]
[[12.33333333 21.33333333]
 [21.33333333 37.33333333]]
Eigen vectors:  [[-0.86762506 -0.49721902]
 [ 0.49721902 -0.86762506]]
Eigen values:  [ 0.10761573 49.55905094]
[[ 0.24024879  6.28473039]
 [-0.37375033  1.32257309]
 [ 0.13350154 -7.60730348]]

 Result:

---------------------------------------------------
