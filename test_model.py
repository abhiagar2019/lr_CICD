import numpy as np
from sklearn.metrics import r2_score
import joblib
import train # This imports our train.py script so we can call its functions
import random

MODEL_FILENAME = "model.joblib"
R2_THRESHOLD = 0.75  # Our target "goodness" score (75%)
MAX_RETRAIN_ATTEMPTS = 2 # Try retraining a maximum of 2 times

def load_model_func(filename=MODEL_FILENAME):
    """Loads the model from a file."""
    try:
        model = joblib.load(filename)
        print(f"Model loaded successfully from {filename}")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file '{filename}' not found. Train the model first.")
        exit(1) # Exit with an error code

def generate_test_data_func(n_samples=50):
    """Generates new, unseen data for testing."""
    print(f"Generating test data with {n_samples} samples.")
    X_test = np.random.rand(n_samples, 1) * 10 # Single feature, values from 0 to 10

    # Simulate a "true" underlying pattern for test data
    # It's good if this is similar to, but not exactly, the training generation
    true_slope_test = 2.0 * random.uniform(0.85, 1.15) # Test on slightly different "true" params
    true_intercept_test = 5.0 * random.uniform(0.85, 1.15)
    noise_level_test = 1.0 * random.uniform(0.8, 1.2)

    y_test = true_slope_test * X_test.squeeze() + true_intercept_test + np.random.randn(n_samples) * noise_level_test
    return X_test.reshape(-1, 1), y_test

def evaluate_model_func(model, X_test, y_test):
    """Evaluates the model and returns the R-squared score."""
    y_pred = model.predict(X_test)
    current_r2 = r2_score(y_test, y_pred)
    return current_r2

def perform_test_cycle(current_test_attempt=1, retraining_count=0):
    """Performs a cycle of testing and potential retraining."""
    print(f"\n--- Starting Model Test Cycle (Test Attempt: {current_test_attempt}, Retrains Done: {retraining_count}) ---")
    model = load_model_func()
    X_test, y_test = generate_test_data_func()

    current_r2 = evaluate_model_func(model, X_test, y_test)
    print(f"R-squared score on current test data: {current_r2:.4f}")

    if current_r2 < R2_THRESHOLD:
        print(f"R-squared score {current_r2:.4f} is BELOW the threshold of {R2_THRESHOLD}.")
        if retraining_count < MAX_RETRAIN_ATTEMPTS:
            print("Attempting to retrain the model...")
            # Call the main function from train.py to retrain the model
            # Pass the current retraining_count to train.py so it knows it's a retraining attempt
            train.main(retraining_attempt_number=retraining_count + 1) 

            # After retraining, we need to re-run the test cycle.
            # We increment retraining_count.
            perform_test_cycle(current_test_attempt=current_test_attempt + 1, retraining_count=retraining_count + 1)
        else:
            print(f"Maximum retraining attempts ({MAX_RETRAIN_ATTEMPTS}) reached.")
            print("Model FAILED to achieve target R-squared score.")
            exit(1) # Exit with an error code, CI will fail
    else:
        print(f"R-squared score {current_r2:.4f} is ABOVE or EQUAL to the threshold of {R2_THRESHOLD}.")
        print("Model PASSED!")
        exit(0) # Exit with a success code, CI will pass

if __name__ == "__main__":
    perform_test_cycle()
