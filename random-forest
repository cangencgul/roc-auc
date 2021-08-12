from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)

falsePositiveRate, truePositiveRate, treshold = roc_curve(y_test, y_pred)
auc_ = auc(falsePositiveRate, truePositiveRate)

plt.figure()
plt.plot(falsePositiveRate, truePositiveRate)
plt.show()
