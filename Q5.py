from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn . datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(30,)))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=20)
'''
print("Train accuracy:", model.evaluate(X_train_scaled, y_train)[1])
print("Test accuracy:", model.evaluate(X_test_scaled, y_test)[1])

# COMMENTS:
# Feature scaling is needed because neural networks work better when input values are in a similar range.
# An epoch is one full pass through the training dataset.
'''

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn . datasets import load_breast_cancer

data = load_breast_cancer()


model2 = DecisionTreeClassifier(criterion="entropy")
model2.fit(X_train, y_train)

print("Train accuracy:", model2.score(X_train, y_train))
print("Test accuracy:", model2.score(X_test, y_test))

from sklearn.metrics import confusion_matrix

# Decision Tree predictions
y_pred_tree = model2.predict(X_test)

print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

# Neural Network predictions
y_pred_nn = model.predict(X_test_scaled)
y_pred_nn = (y_pred_nn > 0.5)

print("Neural Network Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))

# I prefer the Neural Network because it usually gives better accuracy.
# Decision Tree:
# Advantage: easy to understand because you can visualize
# Limitation: prone to overfitting

# Neural Network:
# Advantage: more powerful and accurate
# Limitation: harder to understand because of black box