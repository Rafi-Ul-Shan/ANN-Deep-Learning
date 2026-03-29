import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Load Data (Assumes you have a dataset like weatherAUS.csv)
# Example data structure: https://www.kaggle.com
df = pd.read_csv('weatherAUS.csv')

# 2. Data Preprocessing
df = df.dropna()  # Remove missing values for simplicity
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Select features and target
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
            'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 
            'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
X = df[features]
y = df['RainTomorrow']

# 3. Split and Scale Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build the ANN Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 6. Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 7. Save the model
model.save('rain_ann_model.h5')
