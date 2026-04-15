import bentoml
from sklearn import svm 
from sklearn import datasets

#load training dataset 
iris=datasets.load_iris()
X,y=iris.data, iris.target

#train models 
clf=svm.SVC(gamma="scale")
clf.fit(X,y)

saved_model=bentoml.sklearn.save_model("iris_clf",clf)
print(f"Model saved: {saved_model}")
