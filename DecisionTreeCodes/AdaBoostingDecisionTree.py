import pandas
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as graphPlotter
dataframe = pandas.read_csv("ionosphere.csv")
array = dataframe.values
X = array[:,0:33]
print("array", array)
Y = array[:,34]
kVals = []
meanTrainErrs = []
meanValidationErrs = []
minK = 2
maxK = 10
seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
model = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
for k in range(minK, (maxK + 1)):
  kVals.append(k)
  kfold = model_selection.KFold(n_splits=k, random_state=seed)
  # results = model_selection.cross_val_score(model, X, Y, cv=kfold)
  results = model_selection.cross_validate(model, X, Y, return_train_score=True, cv=kfold)
  print(results['train_score'])
  print(results['test_score'])
  meanTrainErrs.append( 1 - (results['train_score'].mean()) )
  meanValidationErrs.append( 1 - (results['test_score'].mean()) )
  # print(meanTrainErrs)
  # print(meanValidationErrs)
  # print(kVals)
  # print(results)
graphPlotter.plot(kVals, meanTrainErrs)
graphPlotter.plot(kVals, meanValidationErrs)
graphPlotter.title('Train Error, Validation Error vs K-Fold')
graphPlotter.xlabel('K-Fold')
graphPlotter.ylabel('Error')
graphPlotter.show()
