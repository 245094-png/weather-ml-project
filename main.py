import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Program started")

# ===================== LOAD DATA =====================
url = "https://raw.githubusercontent.com/245094-png/weather-ml-project/main/weatherHistory.csv"
data = pd.read_csv(url)

print("Data loaded")

# ===================== DATA PREPROCESSING =====================
data = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Summary']]
data = data.dropna()

data['Summary'] = data['Summary'].astype('category').cat.codes

X = data.drop('Summary', axis=1)
y = data['Summary']

# ===================== VISUALIZATION =====================
plt.figure()
data['Summary'].value_counts().plot(kind='bar')
plt.title("Weather Classes Distribution")
plt.show()

# ===================== SCALING =====================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===================== TRAIN/TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== MODEL 1: KNN =====================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print("KNN Accuracy:", knn_acc)

# ===================== MODEL 2: DECISION TREE =====================
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_acc)

# ===================== MODEL 3: RANDOM FOREST =====================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)

# ===================== MODEL COMPARISON =====================
models = ['KNN', 'Decision Tree', 'Random Forest']
acc = [knn_acc, dt_acc, rf_acc]

plt.figure()
plt.bar(models, acc)
plt.title("Model Comparison")
plt.show()

# ===================== BEST MODEL =====================
best_model = max([
    ("KNN", knn_acc),
    ("Decision Tree", dt_acc),
    ("Random Forest", rf_acc)
], key=lambda x: x[1])

print("Best Model:", best_model[0], "Accuracy:", best_model[1])

# ===================== PREDICTION =====================
temp = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
wind = float(input("Enter Wind Speed: "))

result = rf.predict(
    pd.DataFrame([[temp, humidity, wind]],
                 columns=['Temperature (C)', 'Humidity', 'Wind Speed (km/h)'])
)

print("Predicted Weather:", result[0])
