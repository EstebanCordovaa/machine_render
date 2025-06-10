import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv('data/raw/iris.csv', index=False)
print("¡Dataset Iris guardado en data/raw/iris.csv!")