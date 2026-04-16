import bentoml 
iris_clf_runner=bentoml.sklearn.get("iris_clf:latest").to_runner()
iris_clf_runner.init_local()
print(iris_clf_runner.predict.run([[5.9,3.,5.1,1.8]]))
#C:\Users\ASUS\Downloads\ML_project_end_to_end\MLflow_Dagshub_and_BentoML.MLFlow.Mlflowexperiments-main.app.py
#C:\Users\ASUS\Downloads\ML_project_end_to_end\MLflow_Dagshub_and_BentoML\MLFlow\Mlflowexperiments-main\app.py