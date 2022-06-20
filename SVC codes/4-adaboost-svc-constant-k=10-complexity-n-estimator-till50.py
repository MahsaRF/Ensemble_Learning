import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as graphPlotter
from sklearn.svm import SVC
dataframe = pandas.read_csv("ionosphere.csv")
array = dataframe.values
X = array[:,0:33]
print("array", array)
Y = array[:,34]
seed = 7
numTrees = []
meanTrainErrs = []
meanValidationErrs = []
maxNumTrees = 100
cart = SVC(probability=True,gamma='scale',degree=3)
k = 10
kfold = model_selection.KFold(n_splits=k, random_state=seed)
for numTree in range(1, maxNumTrees + 1):
  numTrees.append(numTree)
  model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=cart, n_estimators=numTree, random_state=seed)
  #results = model_selection.cross_val_score(model, X, Y, cv=kfold)
  results = model_selection.cross_validate(model, X, Y, return_train_score=True, cv=kfold)
  meanTrainErrs.append( 1 - (results['train_score'].mean()) )
  meanValidationErrs.append( 1 - (results['test_score'].mean()) )
  #print(meanTrainErrs)
  #print(meanValidationErrs)
  #print(numTrees)
  # print(results)
graphPlotter.plot(numTrees, meanTrainErrs)
graphPlotter.plot(numTrees, meanValidationErrs)
graphPlotter.title('Train Error, Validation Error vs Number  of  Trees')
graphPlotter.xlabel('Number  of  Trees')
graphPlotter.ylabel('Error')
graphPlotter.show()