import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as graphPlotter
dataframe = pandas.read_csv("ionosphere.csv")
array = dataframe.values
X = array[:,0:33]
# print("array", array)
Y = array[:,34]
seed = 7
numTrees = []
meanTrainErrs = []
meanValidationErrs = []
maxNumTrees = 50
cart = DecisionTreeClassifier()
k = 10
kfold = model_selection.KFold(n_splits=k, random_state=seed)
for numTree in range(1, maxNumTrees + 1):
  numTrees.append(numTree)
  model = BaggingClassifier(base_estimator=cart, n_estimators=numTree, random_state=seed)
  # results = model_selection.cross_val_score(model, X, Y, cv=kfold)
  results = model_selection.cross_validate(model, X, Y, return_train_score=True, cv=kfold)
  # print(results['train_score'])
  # print(results['test_score'])
  meanTrainErrs.append( 1 - (results['train_score'].mean()) )
  meanValidationErrs.append( 1 - (results['test_score'].mean()) )
  # print(meanTrainErrs)
  # print(meanValidationErrs)
  # print(numTrees)
  # print(results)
graphPlotter.plot(numTrees, meanTrainErrs)
graphPlotter.plot(numTrees, meanValidationErrs)
graphPlotter.title('Train Error, Validation Error vs Number  of  Trees')
graphPlotter.xlabel('Number  of  Trees')
graphPlotter.ylabel('Error')
graphPlotter.show()
