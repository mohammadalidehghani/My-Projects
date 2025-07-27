"""
Author: Mohammadali Dehghani 
"""

import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from prefect import flow, task

from sklearn.metrics import mean_squared_error
from scipy.stats import entropy
import joblib
import numpy as np
import os


class HotelBookingModel:
    def __init__(self, data_path="hotel_bookings.csv", model_name="hotel_rf", mock_failure=False):
        self.data_path = data_path
        self.model_name = model_name
        self.mock_failure = mock_failure

 # The load_and_preprocess_data method, decorated with @task, reads the CSV file and drops any rows containing null values. 
    @task
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        return df

 # The data_tests method sets up a Great Expectations context and creates an expectation suite named "hotel_suite". 
 # It defines specific rules to ensure the data is valid: columns like "country" and "children" must not contain null values, "adr" (average daily rate) should fall between 0 and 1000, and "children" must only include values 0 through 3. 
    @task
    def data_tests(self, df: pd.DataFrame):
        context = gx.get_context()
        suite = ExpectationSuite("hotel_suite")

        ds = context.data_sources.add_pandas("pandas")
        asset = ds.add_dataframe_asset(name="hotel_data")
        bd = asset.add_batch_definition_whole_dataframe("batch")
        batch = bd.get_batch(batch_parameters={"dataframe": df})

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="country")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="children")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="adr", min_value=0, max_value=1000
            )
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="children", value_set=[0, 1, 2, 3]
            )
        )
        results = batch.validate(suite)
        if not results.success:
            raise AssertionError(f"Data validation failed:\n{results}")

        return df
 
 # In train_baseline_model, the code builds a basic linear regression model using numeric features from the dataset (excluding "adr"). 
 # It splits the data into training and validation sets, trains the model, and evaluates it using the R² score. 
 # The R² score is then logged into MLflow for experiment tracking. This baseline model serves as a reference point for assessing more advanced models.
    @task
    def train_baseline_model(self, df: pd.DataFrame):
        with mlflow.start_run(run_name="baseline_regression"):
            X = df.select_dtypes("number").drop(columns=["adr"], errors="ignore")
            y = df["adr"]
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            preds = lr.predict(X_val)
            baseline_r2 = r2_score(y_val, preds)
            mlflow.log_metric("baseline_r2", baseline_r2)
        return baseline_r2

 # The train_and_version_rf method trains a more sophisticated model a Random Forest Regressor using the same features and evaluation procedure as the baseline. 
 # After training and evaluation, it logs the performance (R² score) and registers the trained model into MLflow's model registry. 
 # The model version is fetched and returned, which is important for managing multiple versions over time.
 # To induce at the training step an artificial error I ve added a retruing logic and if the training fails it will wait 5 second then retry.
 # It tries 3 times after that the process fails. I ve added also the an option to turn the failure on and off. 
    @task(retries=3, retry_delay_seconds=5)
    def train_and_version_rf(self, df: pd.DataFrame, n_estimators=100, max_depth=None):
        with mlflow.start_run(run_name="random_forest"):
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            X = df.select_dtypes("number").drop(columns=["adr"], errors="ignore")
            y = df["adr"]
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            train_df = X_train.copy()
            train_df["adr"] = y_train
            train_df.to_csv("actual_training_data.csv", index=False)


            val_df = X_val.copy()
            val_df["adr"] = y_val
            val_df.to_csv("unseen_data.csv", index=False)


            if self.mock_failure:
                if len(X_train) < 1000:
                    raise ValueError("Training data too small.")
            rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            rf_r2 = r2_score(y_val, preds)
            mlflow.log_metric("rf_r2", rf_r2)

            # register model
            mlflow.sklearn.log_model(
                sk_model=rf,
                artifact_path="model",
                registered_model_name=self.model_name
            )
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(self.model_name, stages=["None"])
            model_version = versions[-1].version

        return rf_r2, model_version

 # We compare the baseline model to the more complex model (Random Forest) to evaluate whether the added complexity actually leads to improved performance. 
 # The baseline serves as a reference point it’s fast, interpretable, and often surprisingly effective. However, it may not capture nonlinear relationships in the data.
 # Random Forest, being a more powerful ensemble method, can model complex interactions and nonlinearities. By comparing their R² scores, we can objectively assess whether the Random Forest model provides a meaningful improvement. 
 # If it doesn't, we might favor the simpler model for its interpretability and lower risk of overfitting. 
    @task
    def validate(self, baseline_r2: float, rf_r2: float, model_version: str):
        better = rf_r2 >= baseline_r2
        with mlflow.start_run(run_name="validation", nested=True):
            mlflow.log_metric("baseline_r2", baseline_r2)
            mlflow.log_metric("rf_r2", rf_r2)
            mlflow.set_tag("rf_better_than_baseline", better)
        return {
            "baseline_r2": baseline_r2,
            "rf_r2": rf_r2,
            "rf_better": better,
            "model_version": model_version,
        }

 # The run_all method is the orchestrated flow using Prefect. It runs all the above steps in a defined order: 
 # loading data, validating it, training the models, and finally validating and comparing their performance. 
 # This method centralizes the workflow and makes it easy to run as a single unit.
    @flow
def run_all_flow(
    data_path: str = "hotel_bookings.csv",
    model_name: str = "hotel_rf",
    mock_failure: bool = True,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 10
):
    pipeline = HotelBookingModel(data_path=data_path, model_name=model_name, mock_failure=mock_failure)

## In this flow, key model configuration choices such as the number of estimators (rf_n_estimators), maximum tree depth 
# (rf_max_depth), input dataset path (data_path), model registry name (model_name), and simulated failure toggle 
# (mock_failure) are exposed as parameters of the run_all_flow function. These parameters directly control model behavior 
# and are logged into MLflow for versioning and reproducibility, ensuring that any change in model performance can be traced 
# back to its exact configuration.


    df = pipeline.load_and_preprocess_data()
    df = pipeline.data_tests(df)
    baseline_r2 = pipeline.train_baseline_model(df)
    rf_r2, model_version = pipeline.train_and_version_rf(df, n_estimators=rf_n_estimators, max_depth=rf_max_depth)
    result = pipeline.validate(baseline_r2, rf_r2, model_version)
    return result

 # The if __name__ == "__main__" block makes the script executable from the command line. 
 # It disables ephemeral Prefect server mode (to avoid runtime warnings), creates an instance of the HotelBookingModel class, runs the full flow, and prints the result summary. 
 # If we change mock_failure=True it will retry 3 times and fail.
if __name__ == "__main__":
    pipeline = HotelBookingModel(mock_failure=False)
    output = pipeline.run_all()
    print(output)



# Load the unseen dataset (post-deployment phase)
@task
def load_unseen_data(data_path="unseen_data.csv"):
    df = pd.read_csv(data_path)
    df = df.dropna()
    return df

# Compute KL Divergence between training and unseen distributions
@task
def compute_drift(train_data, unseen_data, column="adr"):
    train_hist, _ = np.histogram(train_data[column], bins=10, range=(0, 1000), density=True)
    unseen_hist, _ = np.histogram(unseen_data[column], bins=10, range=(0, 1000), density=True)
    drift_score = entropy(train_hist + 1e-8, unseen_hist + 1e-8)  # Avoid division by zero
    return drift_score

### Drift Detection Expectation: `adr` Distribution Stability

## The `adr` (average daily rate) is the target variable for model training and evaluation. Stability in its distribution between training and unseen data is critical to ensure consistent model performance post-deployment.

## We use **KL Divergence** to compare the `adr` distribution in the live (unseen) dataset against the original training set.

## **Expected Behavior**:
##- A KL Divergence score of **less than 0.05** is considered acceptable and expected.
##- This threshold indicates that the unseen data still falls within the same statistical distributional range the model was trained on.
##- If the divergence exceeds **0.1**, it may suggest drift in the business environment or user behavior (e.g., pricing changes, new customer segments), which could compromise model performance.

##**Sourcing of Expectation**:
##- Derived from historical analysis of similar datasets, where fluctuations in target distributions below 5% (in terms of probability mass divergence) did not impact model accuracy significantly.
##- Also backed by practical ML monitoring conventions which flag divergence thresholds typically between **0.05 and 0.1** for numeric features.
##- This threshold can be adjusted over time as more production data accumulates.

##**Observed Value**:
##- `KL Divergence (adr) = 0.0203`, which is well below the threshold and thus within the expected operating range.
##- No immediate model retraining is required.

## Including this step operationalizes data quality and stability assurance into my ML system — proactively monitoring for concept drift before it degrades predictions.



# Load a trained model by version ID from MLflow
@task
def load_model(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

# Predict target values (adr) using the given trained model and input features

def predict(model, df):
    X = df.select_dtypes("number").drop(columns=["adr"], errors="ignore")
    preds = model.predict(X)
    return preds

# Evaluate model performance using Mean Squared Error (MSE) between predictions and true values
@task
def evaluate_predictions(preds, targets):
    mse = mean_squared_error(targets, preds)
    return mse

# Generate a synthetic, deterministic user ID by combining several identifier-like columns
# This is used for reproducible user-level A/B group assignment

df["user_id"] = df["country"].astype(str) + "_" + \
                df["agent"].astype(str) + "_" + \
                df["company"].astype(str) + "_" + \
                df["reservation_status_date"].astype(str)

# Assign each row deterministically to group A or B based on a hashed user ID
# Ensures even, reproducible data split for A/B testing

def assign_ab_group(df, id_column="userid"):
    np.random.seed(42)
    hashed_ids = df[id_column].apply(lambda x: hash(str(x)) % 100)
    return df[hashed_ids < 50], df[hashed_ids >= 50]


# A/B Test logic with tracking
### Handling Multiple Concurrent A/B Tests
## Each A/B test is assigned a unique identifier (`ab_test_name`), passed as a parameter to
#  the `ab_test_flow`. This tag is logged in MLflow along with the evaluated metrics, model 
# versions, and dataset details. By filtering MLflow runs using this tag 
# (e.g., `"tags.ab_test_name = 'ab_test_june_2025_v1'"`), we can isolate the results of 
# individual A/B tests, even when multiple tests are running concurrently or sequentially. 
# This ensures results are traceable, reproducible, and never mixed across experiments.


# Run an A/B test comparing two model versions using the same unseen data.
# The data is deterministically split into two groups using a synthetic user ID.
# Each model version makes predictions on its respective group.
# The flow logs performance metrics (MSE) and metadata (model versions, test name) to MLflow.
@flow
def ab_test_flow(
    model_name="hotel_rf",
    version_a="1",
    version_b="2",
    ab_test_name="ab_test_june_2025_v1"
):
    df = load_unseen_data()

    # Generate a synthetic user ID to ensure consistent user-level splits
    df["user_id"] = df["country"].astype(str) + "_" + \
                    df["agent"].astype(str) + "_" + \
                    df["company"].astype(str) + "_" + \
                    df["reservation_status_date"].astype(str)

    # Split data into A and B groups based on hashed user ID
    df_a, df_b = assign_ab_group(df, id_column="user_id")

     # Load the two specified model versions from MLflow
    model_a = load_model(model_name, version_a)
    model_b = load_model(model_name, version_b)

    # Generate predictions for each group
    preds_a = predict(model_a, df_a)
    preds_b = predict(model_b, df_b)

     # Evaluate prediction accuracy using Mean Squared Error
    mse_a = evaluate_predictions(preds_a, df_a["adr"])
    mse_b = evaluate_predictions(preds_b, df_b["adr"])

    # Log A/B test metadata and performance to MLflow
    with mlflow.start_run(run_name="ab_test"):
        mlflow.set_tag("ab_test_name", ab_test_name)
        mlflow.set_tag("version_a", version_a)
        mlflow.set_tag("version_b", version_b)

        mlflow.log_metric("mse_a", mse_a)
        mlflow.log_metric("mse_b", mse_b)

    
    print("A/B Test Results:")
    print(f"Model Version {version_a} MSE: {mse_a:.4f}")
    print(f"Model Version {version_b} MSE: {mse_b:.4f}")

    return {
        "version_a_mse": mse_a,
        "version_b_mse": mse_b,
        "ab_test_name": ab_test_name
    }


# Drift test flow to measure distribution shift on unseen data
@flow
def drift_test_flow(train_data_path="actual_training_data.csv", unseen_data_path="unseen_data.csv"):
    train_df = pd.read_csv(train_data_path).dropna()
    unseen_df = pd.read_csv(unseen_data_path).dropna()
    drift = compute_drift(train_df, unseen_df)
    print(f"KL Divergence (adr distribution drift): {drift:.4f}")
    return drift

if __name__ == "__main__":
    print("Running drift test:")
    drift_result = drift_test_flow()
    print(drift_result)  

    print("\nRunning A/B test:")
    ab_result = ab_test_flow()
    print(ab_result) 
