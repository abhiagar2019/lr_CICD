# python

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import random # To add some variability

MODEL_FILENAME = "model.joblib" # Standard name for our saved model file

def generate_data(n_samples=100, is_retraining_attempt=False):
    """Generates some simple data for linear regression."""
    print(f"Generating data with {n_samples} samples. Retraining attempt: {is_retraining_attempt}")
    X = np.random.rand(n_samples, 1) * 10  # Single feature, values from 0 to 10

    # Let's make the "true" relationship slightly variable
    # This way, retraining might actually produce a different model
    if is_retraining_attempt:
        # For retraining, maybe we simulate getting slightly "cleaner" data or more data
        n_samples = int(n_samples * 1.5) # More data
        noise_level = 0.8 * random.uniform(0.8, 1.2) # Slightly less noise on average
        true_slope = 2.0 * random.uniform(0.9, 1.1) # slope around 2
        true_intercept = 5.0 * random.uniform(0.9, 1.1) # intercept around 5
    else:
        noise_level = 1.0 * random.uniform(0.8, 1.2)
        true_slope = 2.0 * random.uniform(0.9, 1.1)
        true_intercept = 5.0 * random.uniform(0.9, 1.1)
    
    # y = slope * X + intercept + some_random_noise
    y = true_slope * X.squeeze() + true_intercept + np.random.randn(n_samples) * noise_level
    
    # Reshape X to be a 2D array for scikit-learn
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

def main(retraining_attempt_number=0): # Default is 0, meaning initial training
    """Main function to run the training process."""
    print(f"\n--- Starting Model Training (Overall Attempt: {retraining_attempt_number + 1}) ---")
    
    is_retraining = retraining_attempt_number > 0
    X_train, y_train = generate_data(n_samples=100, is_retraining_attempt=is_retraining)
    
    model = train_model_func(X_train, y_train)
    save_model_func(model)
    
    # Optional: Evaluate on training data itself (just for info)
    y_pred_train = model.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)
    print(f"R-squared score on training data: {train_r2:.4f}")
    print("--- Model Training Finished ---")

if __name__ == "__main__":
    # This code runs if you execute "python train.py" directly
    main()
