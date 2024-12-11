import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.info())
print(train.isnull().sum())
sns.countplot(x='Survived', data=train)
plt.title('Survivors vs Non-Survivors')
plt.show()
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train['FamilySize'] = train['SibSp'] + train['Parch']
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_val, y_pred))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test['FamilySize'] = test['SibSp'] + test['Parch']
test_predictions = model.predict(test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)
