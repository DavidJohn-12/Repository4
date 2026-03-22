from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn . datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# Feature importance
import numpy as np
importance = model.feature_importances_

sorted_indices = np.argsort(importance)

top5 = sorted_indices[-5:]

print(top5)
print("Top 5 important features:", top5)

# COMMENTS:
# Limiting tree depth reduces overfitting so simpler models generalize better to new data.
# Feature importance shows which features are most useful, making the model easier to understand patterns.