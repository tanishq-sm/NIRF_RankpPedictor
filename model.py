import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv(r"C:\\Users\\VICTUS\\OneDrive\\Desktop\\codeutsava\\codeUtsavNIRF2023\\OverallRanking_2017.csv")  # Replace 'your_dataset.csv' with your dataset file
# Prepare your data
X = data[['TLR', 'RPC', 'GO', 'OI', 'Perception']]
y = data['Rank']  # Assuming 'NIRF_Rank' is the column containing the target variable

# Split the dataset into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest regression model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters
rf_regressor.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = rf_regressor.predict(X_test) #modified using ceiling function
print(y_pred)
print(y_test)
# Calculate regression evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2) Score: {r2}")

# Create a scatter plot of actual vs. predicted values
#plt.figure(figsize=(8, 6))
#plt.plot(y_test, y_pred, c='blue', marker='o', alpha=0.5)
#plt.xlabel('Actual Values (NIRF Rank)')
#plt.ylabel('Predicted Values')
#plt.title('Actual vs. Predicted Values')
#plt.show()

#TLR_value=85;
#RPC_value=68;
#GO_value=93;
#OI_value=58;
#Perception=47;
# Assuming you have a list of values for TLR, RPC, GO, and OI
new_data_point = [TLR_value, RPC_value, GO_value, OI_value , Perception]

# Use the trained Random Forest regressor to make a prediction
predicted_nirf_rank = math.ceil(rf_regressor.predict([new_data_point]))

#print("Predicted NIRF Rank: ",predicted_nirf_rank)


import pickle
pickle.dump(rf_regressor,open("model2017.pkl","wb"))
