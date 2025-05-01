import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Load dataset
df = pd.read_csv("synthetic_rainfall_data.csv")

# Convert date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Extract season from the date
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["Season"] = df["Date"].apply(get_season)

# Feature Engineering - Adding lagged rainfall for season-based prediction
df.sort_values(by=["City", "Date"], inplace=True)
df["Lag_1"] = df.groupby("City")["Rainfall"].shift(1)
df["Lag_2"] = df.groupby("City")["Rainfall"].shift(2)
df["Lag_3"] = df.groupby("City")["Rainfall"].shift(3)
df.dropna(inplace=True)

# Feature scaling
scaler = MinMaxScaler()
features = ["Temperature", "Humidity", "Pressure", "Wind Speed", "Cloud Cover", "Lag_1", "Lag_2", "Lag_3"]
target = "Rainfall"
df[features] = scaler.fit_transform(df[features])

# Prepare seasonal data
def get_seasonal_data(city, input_date):
    season = get_season(pd.to_datetime(input_date))
    subset = df[(df["City"] == city) & (df["Season"] == season)]
    X = subset[features].values
    y = subset[target].values.reshape(-1, 1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Rainfall classification based on mm/hr
def classify_rainfall(mm_per_hour):
    if mm_per_hour < 2.5:
        return "🌦️ No"
    elif mm_per_hour < 10:
        return "🌧️ Moderate"
    elif mm_per_hour < 50:
        return "🌩️ Heavy"
    else:
        return "⛈️ Violent"

# Neural Network Model
class RainfallPINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RainfallPINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
# PINN loss
def physics_informed_loss(model, X, y_true):
    y_pred = model(X)
    mse_loss = nn.MSELoss()(y_pred, y_true)

    X.requires_grad_(True)
    y_phys = model(X)
    gradients = torch.autograd.grad(outputs=y_phys, inputs=X,
                                    grad_outputs=torch.ones_like(y_phys),
                                    create_graph=True)[0]

    grad_pressure = gradients[:, features.index("Pressure")]
    grad_wind = gradients[:, features.index("Wind Speed")]
    navier_stokes_reg = torch.mean(grad_pressure**2 + grad_wind**2)

    total_loss = mse_loss + 0.05 * navier_stokes_reg
    return total_loss, mse_loss.item(), navier_stokes_reg.item()

# Train Model
def train_model(X_train, y_train):
    input_size, hidden_size, output_size = len(features), 128, 1
    model = RainfallPINN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    for epoch in range(2000):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return model

# Evaluate Model
def calculate_accuracy(model, X_test, y_test):
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        mse = mean_squared_error(y_test.numpy(), y_pred)
        r2 = r2_score(y_test.numpy(), y_pred)
    return mse, r2, y_pred

# Streamlit UI
st.set_page_config(page_title="Rainfall Prediction", layout="wide")
st.title("🌧️ Rainfall Prediction System")

st.sidebar.header("User Input Parameters")
city = st.sidebar.selectbox("Select City", df["City"].unique())
input_date = st.sidebar.date_input("Select Date")

if st.sidebar.button("Predict Next 7 Days"):
    X_train, y_train = get_seasonal_data(city, str(input_date))
    model = train_model(X_train, y_train)
    X_future = torch.tensor(X_train[-7:], dtype=torch.float32)
    predictions = model(X_future).detach().numpy()

    mse, r2, y_pred = calculate_accuracy(model, X_train, y_train)

    # Build forecast DataFrame with classification
    forecast_dates = pd.date_range(start=input_date, periods=7)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Predicted Rainfall (mm)": predictions.flatten()
    })
    forecast_df["Rain Intensity"] = forecast_df["Predicted Rainfall (mm)"].apply(classify_rainfall)

    # Display Predictions
    st.subheader("📊 Predicted Rainfall for Next 7 Days")
    st.dataframe(forecast_df.style.format({
        "Predicted Rainfall (mm)": "{:.2f}"
    }))

    # # Accuracy Metrics
    # st.subheader("📈 Model Performance")
    # st.write(f"**MSE:** {mse:.4f}")
    # st.write(f"**R² Score:** {r2:.4f}")

    # Optional: Show success message
    st.success("✅ Prediction Completed!")
