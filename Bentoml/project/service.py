import numpy as np 
import bentoml
from bentoml.io import NumpyNdarray
iris_clf_runner=bentoml.sklearn.get("iris_clf:latest").to_runner()
svc=bentoml.Service("iris_classifier",runners=[iris_clf_runner])
@svc.api(input=NumpyNdarray(),output=NumpyNdarray())
def classify(input_series:np.ndarray)->np.ndarray:
    result=iris_clf_runner.predict.run(input_series)
    return result
'''import bentoml
import pandas as pd
from bentoml.io import PandasDataFrame

# Load model
model_runner = bentoml.sklearn.get("phishing_model:latest").to_runner()

# Create service
svc = bentoml.Service("phishing_service", runners=[model_runner])

@svc.api(input=PandasDataFrame(), output=PandasDataFrame())
def predict(input_df: pd.DataFrame) -> pd.DataFrame:
    result = model_runner.predict.run(input_df)
    return pd.DataFrame(result, columns=["prediction"])'''
    