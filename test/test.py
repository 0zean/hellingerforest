import numpy as np
import pandas as pd
from hellingerforest import RandomForestClassifier as hf

data = pd.read_csv("test\creditcard.csv")

X = np.array(data.iloc[260000:, 1:30])
y = np.array(data.iloc[260000:, 30]).astype(float)

print("Feature:\n", X[:10])
print("\nTarget:\n", y[:10])

print("\n Shape of X:", X.shape)

model = hf(n_estimators=10, max_depth=7, min_samples_split=10)
model.fit(X, y)

print("Model trained")

print("Predicting...")
predictions = model.predict(X)
print("Predictions:\n", predictions)

print("\nUnique predictions:", np.unique(predictions))

print(np.where(np.array(predictions) > 0))
print(np.where(y > 0))
