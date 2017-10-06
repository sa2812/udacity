#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier
import sklearn.ensemble as ensemble 

## k Nearest Neighbors

# print 'k-NN\n-------'
# clf_kNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
# t0_kNN = time()
# clf_kNN.fit(features_train, labels_train)
# print 'fitting: '+str(time()-t0_kNN)

# t1_kNN = time()
# pred_kNN = clf_kNN.predict(features_test)
# print 'predicting: '+str(time()-t1_kNN)

# print clf_kNN.score(features_test, labels_test)

# try:
# 	prettyPicture(clf_kNN, features_test, labels_test)
# except NameError:
# 	pass

## Adaboost

print 'Adaboost\n-------'
clf_ada = ensemble.AdaBoostClassifier(base_estimator=ensemble.RandomForestClassifier(n_estimators=15, min_samples_split=4, max_depth=3))
t0_ada = time()
clf_ada.fit(features_train, labels_train)
print 'fitting: '+str(time()-t0_ada)

t1_ada = time()
pred_ada = clf_ada.predict(features_test)
print 'predicting: '+str(time()-t1_ada)

print clf_ada.score(features_test, labels_test)

try:
	prettyPicture(clf_ada, features_test, labels_test)
except NameError:
	pass

## Random Forest

print 'Random Forest\n-------'
clf_rf = ensemble.RandomForestClassifier(n_estimators=15, min_samples_split=4, max_depth=3)
t0_rf = time()
clf_rf.fit(features_train, labels_train)
print 'fitting: '+str(time()-t0_rf)

t1_rf = time()
pred_rf = clf_rf.predict(features_test)
print 'predicting: '+str(time()-t1_rf)

print clf_rf.score(features_test, labels_test)

try:
    prettyPicture(clf_rf, features_test, labels_test)
except NameError:
    pass
