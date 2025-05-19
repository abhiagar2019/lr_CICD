# train.py

import numpy as np
from sklearn.linear_model import LinearRegression # Make sure all imports are at the top
from sklearn.metrics import r2_score
import joblib
import random

MODEL_FILENAME = "model.joblib"

def generate_data(n_samples_initial=100, is_retraining_attempt=False): # Changed n_samples to n_samples_initial for clarity
    """Generates some simple data for linear regression."""

    actual_n_samples = n_samples_initial

    if is_retraining_attempt:
        actual_n_samples = int(n_samples_initial * 1.5) # More data for retraining
        noise_reduction_factor = 0.8 # Simulate slightly cleaner data
        print(f"Retraining attempt: Generating {actual_n_samples} samples, noise factor {noise_reduction_factor}.")
    else:
        noise_reduction_factor = 1.0
        print(f"Initial training: Generating {actual_n_samples} samples, noise factor {noise_reduction_factor}.")

    X = np.random.rand(actual_n_samples, 1) * 10  # Single feature, values from 0 to 10

    true_slope = random.uniform(1.8, 2.2) 
    true_intercept = random.uniform(4.5, 5.5) 
    base_noise_level = random.uniform(0.8, 1.2) * noise_reduction_factor

    # y = slope * X + intercept + some_random_noise
    # --- THIS IS THE KEY FIX ---
    # Ensure noise array has the same number of samples as X
    noise = np.random.randn(X.shape[0]) * base_noise_level # Use X.shape[0] for noise size

    y = true_slope * X.squeeze() + true_intercept + noise

    return X.reshape(-1, 1), y

def train_model_func(X_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def save_model_func(model, filename=MODEL_FILENAME):
    """Saves the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def main(retraining_attempt_number=0):
    """Main function to run the training process."""
    print(f"\n--- Starting Model Training (Overall Attempt: {retraining_attempt_number + 1}) ---")

    is_retraining = retraining_attempt_number > 0
    # Pass the base number of samples to generate_data
    X_train, y_train = generate_data(n_samples_initial=100, is_retraining_attempt=is_retraining) 

    model = train_model_func(X_train, y_train)
    save_model_func(model)

    y_pred_train = model.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)
    print(f"R-squared score on training data: {train_r2:.4f}")
    print("--- Model Training Finished ---")

if __name__ == "__main__":
    main()
