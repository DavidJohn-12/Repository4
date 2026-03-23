from sklearn . datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

X = data.data
y = data.target

# Shapes
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Class counts:", np.bincount(y))

import pandas as pd
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

print(df.head())

# The dataset is a bit imbalanced.
# Class balance matters because if one class is more significant,
# the model may just predict that class and still get high accuracy.