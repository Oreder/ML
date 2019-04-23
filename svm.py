from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
 
topleft = np.random.uniform(size=(100,2)) + np.array([-1.0, 1.0])
topright = np.random.uniform(size=(100,2)) + np.array([1.0, 1.0])
botleft = np.random.uniform(size=(100,2)) + np.array([-1.0, -1.0])
botright = np.random.uniform(size=(100,2)) + np.array([1.0, -1.0])
 
plt.figure()
plt.scatter(topleft[:,0],topleft[:,1], color='r', marker='o', alpha=0.5, label="class #1")
plt.scatter(topright[:,0],topright[:,1], color='b', marker='s', alpha=0.5, label="class #2")
plt.scatter(botright[:,0],botright[:,1], color='r', marker='o', alpha=0.5)
plt.scatter(botleft[:,0],botleft[:,1], color='b', marker='s', alpha=0.5)
plt.legend()
 
X = np.vstack([topleft, topright, botright, botleft])
Y = np.hstack([[1]*len(topleft), [-1]*len(topright), [1]*len(botright), [-1]*len(botleft)])
 
(trainData, testData, trainLabels, testLabels) = train_test_split(X, Y, test_size=0.25, random_state=42)
 
print("[RESULT] SVM with Linear Kernel")
model = SVC(kernel="linear")
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData)))
 
print("[RESULTS] SVM with Polynomial Kernel")
model = SVC(kernel="poly", degree=2, coef0=1)
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData)))
 
plt.show()