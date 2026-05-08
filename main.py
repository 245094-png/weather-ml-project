import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

print("Program started")

# Load dataset
url = "https://raw.githubusercontent.com/245094-png/weather-ml-project/main/weatherHistory.csv"
data = pd.read_csv(url)

print("Data loaded")

# Select columns
data = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Summary']]
data = data.dropna()

# Convert labels to numbers
data['Summary'] = data['Summary'].astype('category').cat.codes

# Features and target
X = data.drop('Summary', axis=1)
y = data['Summary']

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= KNN =================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print("KNN Accuracy:", knn_acc)

# ================= Decision Tree =================
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_acc)

# ================= Random Forest =================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)

models = ['KNN', 'Decision Tree', 'Random Forest']
accuracies = [knn_acc, dt_acc, rf_acc]

plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()

best_model = max({
    "KNN": knn_acc,
    "Decision Tree": dt_acc,
    "Random Forest": rf_acc
}, key={
    "KNN": knn_acc,
    "Decision Tree": dt_acc,
    "Random Forest": rf_acc
}.get)

print("Best Model:", best_model)
