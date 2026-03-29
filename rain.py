# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # ===============================
# # 1. Generate Synthetic Dataset
# # ===============================

# np.random.seed(42)
# rows = 2000

# temperature = np.random.uniform(15, 45, rows)
# humidity = np.random.uniform(30, 100, rows)
# pressure = np.random.uniform(980, 1050, rows)
# wind_speed = np.random.uniform(0, 20, rows)

# # Rain logic
# rain = ((humidity > 70) & (pressure < 1005) & (temperature < 35)).astype(int)

# data = pd.DataFrame({
#     "temperature": temperature,
#     "humidity": humidity,
#     "pressure": pressure,
#     "wind_speed": wind_speed,
#     "rain": rain
# })

# print("Dataset Generated ✅")

# # ===============================
# # 2. Prepare Data
# # ===============================

# X = data.drop("rain", axis=1)
# y = data["rain"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # ===============================
# # 3. Build ANN Model
# # ===============================

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
#     tf.keras.layers.Dense(8, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# print("Training Started 🚀")

# model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=16,
#     validation_split=0.2,
#     verbose=1
# )

# # ===============================
# # 4. Evaluate Model
# # ===============================

# loss, accuracy = model.evaluate(X_test, y_test)
# print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# # ===============================
# # 5. Prediction Function
# # ===============================

# def predict_rain(temp, hum, pres, wind):
#     input_data = scaler.transform([[temp, hum, pres, wind]])
#     prediction = model.predict(input_data)
    
#     if prediction[0][0] > 0.5:
#         return "Rain Expected 🌧"
#     else:
#         return "No Rain ☀"

# # Example Prediction
# print("\nSample Prediction:")
# print(predict_rain(30, 85, 995, 10))




#______________________
#_________________________
#____________________________\
#_________________________________

# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # ===============================
# # 1. Generate Dataset
# # ===============================
# @st.cache_resource
# def train_model():

#     np.random.seed(42)
#     rows = 2000

#     temperature = np.random.uniform(15, 45, rows)
#     humidity = np.random.uniform(30, 100, rows)
#     pressure = np.random.uniform(980, 1050, rows)
#     wind_speed = np.random.uniform(0, 20, rows)

#     rain = ((humidity > 70) & (pressure < 1005) & (temperature < 35)).astype(int)

#     data = pd.DataFrame({
#         "temperature": temperature,
#         "humidity": humidity,
#         "pressure": pressure,
#         "wind_speed": wind_speed,
#         "rain": rain
#     })

#     X = data.drop("rain", axis=1)
#     y = data["rain"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)

#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
#         tf.keras.layers.Dense(8, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

#     return model, scaler


# model, scaler = train_model()

# # ===============================
# # 2. Streamlit GUI
# # ===============================

# st.title("🌧 Rain Prediction using ANN")
# st.write("Enter weather details to predict rain")

# temp = st.slider("Temperature (°C)", 15.0, 45.0, 30.0)
# hum = st.slider("Humidity (%)", 30.0, 100.0, 70.0)
# pres = st.slider("Pressure (hPa)", 980.0, 1050.0, 1000.0)
# wind = st.slider("Wind Speed (km/h)", 0.0, 20.0, 5.0)

# if st.button("Predict"):

#     input_data = scaler.transform([[temp, hum, pres, wind]])
#     prediction = model.predict(input_data)

#     if prediction[0][0] > 0.5:
#         st.success("🌧 Rain Expected")
#     else:
#         st.info("☀ No Rain Expected")



#________-
#_____________
#_________________
#_________________________
#______________________________

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===================================
# 1. Train Model (Balanced Dataset)
# ===================================

@st.cache_resource
def train_model():

    np.random.seed()  # remove fixed seed for variation
    rows = 2000

    temperature = np.random.uniform(15, 45, rows)
    humidity = np.random.uniform(30, 100, rows)
    pressure = np.random.uniform(980, 1050, rows)
    wind_speed = np.random.uniform(0, 20, rows)

    # 🔥 Add noise for realistic behavior
    noise = np.random.rand(rows)

    rain = (
        (humidity > 65) &
        (pressure < 1008) &
        (temperature < 37) &
        (noise > 0.3)
    ).astype(int)

    data = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "rain": rain
    })

    X = data.drop("rain", axis=1)
    y = data["rain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=0)

    return model, scaler


model, scaler = train_model()

# ===================================
# 2. Streamlit GUI
# ===================================

st.title("🌧 Rain Prediction using ANN (Improved)")
st.write("Enter weather details to predict rain probability")

temp = st.slider("Temperature (°C)", 15.0, 45.0, 30.0)
hum = st.slider("Humidity (%)", 30.0, 100.0, 70.0)
pres = st.slider("Pressure (hPa)", 980.0, 1050.0, 1000.0)
wind = st.slider("Wind Speed (km/h)", 0.0, 20.0, 5.0)

if st.button("Predict"):

    input_data = scaler.transform([[temp, hum, pres, wind]])
    prediction = model.predict(input_data)

    prob = prediction[0][0]

    st.write("### 🌦 Rain Probability:", round(prob * 100, 2), "%")

    if prob > 0.5:
        st.success("🌧 Rain Expected")
    else:
        st.info("☀ No Rain Expected")
