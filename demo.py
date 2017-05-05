from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score # Get accuracy of each classifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1 - K Nearest Neighbor Classifier
clf1 = KNeighborsClassifier(n_neighbors=3)
# 2 - Logistic Regression
clf2 = LogisticRegression()
# 3 - Naive Bayes Classifier
clf3 = GaussianNB()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
# DTC
clf = clf.fit(X, Y)
# 1 - K Nearest Neighbor Classifier
clf1 = clf1.fit(X, Y)
# 2 - Logistic Regression
clf2 = clf2.fit(X, Y)
# 3 - Naive Bayes
clf3 = clf3.fit(X, Y)

# Test data
X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

prediction = clf.predict(X_test)
prediction_1 = clf1.predict(X_test)
prediction_2 = clf2.predict(X_test)
prediction_3 = clf3.predict(X_test)

# CHALLENGE compare their results and print the best one!

print ("Accuracy score for DTC: ", accuracy_score(Y_test, prediction)) # Decision tree classifier
print ("Accuracy score for KNNC: ", accuracy_score(Y_test, prediction_1)) # 1. K Nearest Neighbor Classifier
print ("Accuracy score for LR: ", accuracy_score(Y_test, prediction_2)) # 2. LR
print ("Accuracy score for NB: ", accuracy_score(Y_test, prediction_3)) # 3. NB
