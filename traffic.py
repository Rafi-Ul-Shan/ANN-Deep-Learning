
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

# =====================================
# 1️⃣ Generate Synthetic Dataset
# =====================================

#3000 different scenarios generate kiye.
rows = 3000

hour = np.random.randint(0, 24, rows)  
day = np.random.randint(0, 7, rows)
temperature = np.random.uniform(15, 45, rows)
rain = np.random.randint(0, 2, rows)
holiday = np.random.randint(0, 2, rows)

traffic = []

for i in range(rows):

    score = 0

    # Rush hours  Agar rush hour (7-10 am ya 4-7 pm) hai, to traffic ka score barh jaye.
    if 7 <= hour[i] <= 10 or 16 <= hour[i] <= 19:
        score += 2

    # Weekdays
    if day[i] < 5:
        score += 1

    # Rain increases congestion Agar baarish (rain) hai, to traffic barh jaye.
    if rain[i] == 1:
        score += 1

    # Holiday reduces traffic Agar chutti (holiday) hai, to traffic kam ho jaye.
    if holiday[i] == 1:
        score -= 1

    if score <= 1:
        traffic.append(0)  # Low
    elif score == 2:
        traffic.append(1)  # Medium
    else:
        traffic.append(2)  # High

data = pd.DataFrame({
    "hour": hour,
    "day": day,
    "temperature": temperature,
    "rain": rain,
    "holiday": holiday,
    "traffic": traffic
})

# =====================================
# 2️⃣ Prepare Data
# =====================================
#X hamare features hain (hour, day, etc.)   y hamara target hai (traffic)
X = data.drop("traffic", axis=1)
y = data["traffic"]

#Isay One-Hot Encoding kehte hain
y = to_categorical(y, 3) #Neural network seedha 0, 1, 2 nahi samajhta, wo categories ko format mein chahta hai (e.g., 2 ban jata hai [0, 0, 1]).

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

scaler = StandardScaler()  #sab numbers ko aik hi scale mien leny k liye 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# 3️⃣ Build ANN Model
# =====================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',  #Yeh model ki ghaltiyon ko theek karne wala algorithm hai.
    loss='categorical_crossentropy',  #Yeh measure karta hai ke model kitna ghalat hai./Crossentropy probability difference measure karti hai.
    metrics=['accuracy']
)

print("Training Model...")
model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=1) #epochs=40: Model 40 martaba poore data ko parhay ga aur seekhay ga.

# =====================================
# 4️⃣ Evaluate Model
# =====================================

loss, accuracy = model.evaluate(X_test, y_test)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# =====================================
# 5️⃣ Manual Prediction
# =====================================

print("\n--- Traffic Prediction ---")

hour_input = int(input("Enter Hour (0-23): "))
day_input = int(input("Enter Day (0=Mon, 6=Sun): "))
temp_input = float(input("Enter Temperature (°C): "))
rain_input = int(input("Rain? (0=No, 1=Yes): "))
holiday_input = int(input("Holiday? (0=No, 1=Yes): "))

input_data = scaler.transform([[hour_input, day_input, temp_input, rain_input, holiday_input]])
prediction = model.predict(input_data)

class_index = np.argmax(prediction) #Yeh us category ka index nikalta hai jiski probability sabse high ho.
prob = prediction[0][class_index] * 100

if class_index == 0:
    print(f"\n🟢 Low Traffic ({prob:.2f}%)")
elif class_index == 1:
    print(f"\n🟡 Medium Traffic ({prob:.2f}%)")
else:
    print(f"\n🔴 High Traffic ({prob:.2f}%)")
























