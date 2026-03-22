from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn . datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# COMMENTS:
# Entropy measures how uncertain the data is for partitioning.
# The tree splits data to reduce entropy(better partitioning).
# The training accuracy is not so much higher than test accuracy,the model is not overfitting, thus good generalization.