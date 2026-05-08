import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

print("Program started")

url = "https://raw.githubusercontent.com/245094-png/weather-ml-project/main/weatherHistory.csv"
data = pd.read_csv(url)

print("Data loaded")

data = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Summary']]
data = data.dropna()

data['Summary'] = data['Summary'].astype('category').cat.codes

X = data.drop('Summary', axis=1)
y = data['Summary']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= KNN =================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("KNN Accuracy:", knn_accuracy)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

dt_predictions = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("Decision Tree Accuracy:", dt_accuracy)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)

accuracies = {
    "KNN": knn_accuracy,
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy
}

best_model = max(accuracies, key=accuracies.get)

print("Best Model:", best_model)
