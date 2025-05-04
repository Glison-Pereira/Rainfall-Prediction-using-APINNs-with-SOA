
# Rainfall-Prediction-using-Augmented-Physics-Informed-Neural-Networks-with-SOA
Rainfall Prediction using Augmented Physics-Informed Neural Networks with Season Optimization Algorithm is a rainfall prediction system that predicts rainfall for the next 7 days based on city and date parameters provided by the user. It uses Augmented Physics Informed Neural Networks (APINNs) and Seasonal Optimization Algorithm (SOA).

### Augmented Physics Informed Neural Networks (APINNs)
Augmented Physics Informed Neural Networks (APINNs) enhance Physics Informed Neural Networks by integrating additional domain knowledge into the loss function. This improves the accuracy, especially with noisy or limited data. 

### Seasonal Optimization Algorithm (SOA)
Seasonal Optimization Algorithm (SOA) trains data by season to capture distinct weather patterns. It filters data by city and season for context-aware training in our model.

### 📌 Features

* Seasonal filtering to enhance model generalization
* Augmented PINN model with physics-based loss
* Forecast rainfall for the next 7 days
* Rain intensity classification (No, Moderate, Heavy, Violent)
* Interactive UI built with Streamlit
* Performance metrics (MSE, R²) included for evaluation

### 📂 Dataset

* **File**: `synthetic_rainfall_data.csv`
* **Fields**:

  * `Date`, `City`, `Rainfall`, `Temperature`, `Humidity`, `Pressure`, `Wind Speed`, `Cloud Cover`
* Synthetic data is used to simulate realistic rainfall behavior across multiple cities.


### 🧠 Model Architecture

* 4-layer fully connected neural network (`ReLU` activations)
* Loss function combines:

  * Mean Squared Error (MSE)
  * Navier-Stokes-inspired regularization (∇Pressure² + ∇Wind Speed²)
* Optimizer: `Adam` with learning rate scheduling


### 🔧 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/Rainfall-Prediction-using-APINNs-with-SOA.git
   cd Rainfall-Prediction-using-APINNs-with-SOA
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install torch pandas numpy scikit-learn streamlit matplotlib seaborn
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

### 🖥️ Technologies Used

* Python
* PyTorch
* Streamlit
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn


### 📊 Output Example

| Date       | Predicted Rainfall (mm) | Rain Intensity |
| ---------- | ----------------------- | -------------- |
| 2025-05-04 | 3.24                    | 🌧️ Moderate   |
| 2025-05-05 | 11.80                   | 🌩️ Heavy      |


### 🧪 Performance Metrics

* Mean Squared Error (MSE)
* R² Score

(Shown in app after prediction)


### 📁 File Structure

```
├── app.py                     # Main Streamlit application
├── synthetic_rainfall_data.csv
└── venv/                      # Python virtual environment
```
