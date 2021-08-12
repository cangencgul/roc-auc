from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

clf = SVC()
clf.fit(X_train, y_train)

y_pred = clf.decision_function(X_test)

falsePositiveRate, truePositiveRate, treshold = roc_curve(y_test, y_pred)
auc_ = auc(falsePositiveRate, truePositiveRate)

plt.figure()
plt.plot(falsePositiveRate, truePositiveRate)
plt.show()
