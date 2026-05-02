import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
