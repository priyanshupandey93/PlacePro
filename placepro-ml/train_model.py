import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("placement_dataset.csv")

le = LabelEncoder()
df['work_experience'] = le.fit_transform(df['work_experience'])
df['skills'] = le.fit_transform(df['skills'])
df['extra_curricular'] = le.fit_transform(df['extra_curricular'])
df['placed'] = le.fit_transform(df['placed'])

X = df[['ssc_percentage', 'hsc_percentage', 'degree_percentage',
        'work_experience', 'test_score', 'skills', 'internships',
        'communication_skills', 'extra_curricular']]
y = df['placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

with open("placement_model.pkl", "wb") as f:
    pickle.dump(model, f)