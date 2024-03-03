import argparse
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from category_encoders import CatBoostEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import ast
from sklearn.neural_network import MLPClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--rfc_n_estimators", type=int, default=100)
    parser.add_argument("--rfc_min_samples_split", type=float, default=0.05)
    parser.add_argument("--rfc_criterion", type=str, default="gini")
    parser.add_argument("--logistic_max_iter", type=int, default=500)
    parser.add_argument("--svm_kernel", type=str, default="rbf")
    parser.add_argument("--xgb_max_depth", type=int, default=3)
    parser.add_argument("--nn_hidden_layer_sizes", type=str, default="(100,)")
    args, _ = parser.parse_known_args()
    
    # Read the data
    train_df = pd.read_csv('s3://sagemaker-us-east-1-635439539142/akash/bank-prediction/train_data_bank_pred.csv')  # Path to your train data file
    test_df = pd.read_csv('s3://sagemaker-us-east-1-635439539142/akash/bank-prediction/test_data_bank_pred.csv')   # Path to your test data file
    
    X_train = train_df.drop("y", axis=1)
    y_train = train_df["y"]
    X_test = test_df.drop("y", axis=1)
    y_test = test_df["y"] 
    
    def data_type(dataset):
        """
        Function to identify the numerical and categorical data columns
        :param dataset: Dataframe
        :return: list of numerical and categorical columns
        """
        numerical = []
        categorical = []
        for i in dataset.columns:
            if dataset[i].dtype == 'int64' or dataset[i].dtype == 'float64':
                numerical.append(i)
            else:
                categorical.append(i)
        return numerical, categorical

    numerical, categorical = data_type(X_train)
    
    # Identifying the binary columns and ignoring them from scaling
    def binary_columns(df):
        """
        Generates a list of binary columns in a dataframe.
        """
        binary_cols = []
        for col in df.select_dtypes(include=['int', 'float']).columns:
            unique_values = df[col].unique()
            if np.in1d(unique_values, [0, 1]).all():
                binary_cols.append(col)
        return binary_cols

    binary_cols = binary_columns(X_train)

    # Remove the binary columns from the numerical columns
    numerical = [i for i in numerical if i not in binary_cols]
    
    # Define your encoder
    ct = ColumnTransformer([
        ("CatBoostEncoding", CatBoostEncoder(), categorical),
        ("Scaling", StandardScaler(), numerical)
    ])
    
    # Define classifiers
    rfc = RandomForestClassifier(n_estimators=args.rfc_n_estimators, 
                                  min_samples_split=args.rfc_min_samples_split, 
                                  criterion=args.rfc_criterion)
    
    logistic = LogisticRegression(penalty='l2', max_iter=args.logistic_max_iter)
    
    svm_rbf = SVC(kernel=args.svm_kernel)
    
    xgb_classifier = xgb.XGBClassifier(max_depth=args.xgb_max_depth)

    # Convert hidden_layer_sizes from string to tuple of integers
    hidden_layer_sizes = ast.literal_eval(args.nn_hidden_layer_sizes)
    
    nn_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    
    # Create pipelines for classifiers with your encoder
    rfc_pipeline = Pipeline([
        ("Data Transformations", ct),
        ("Random Forest", rfc)
    ])

    logistic_pipeline = Pipeline([
        ("Data Transformations", ct),
        ("Logistic Regression", logistic)
    ])

    svm_rbf_pipeline = Pipeline([
        ("Data Transformations", ct),
        ("SVM with RBF kernel", svm_rbf)
    ])
    
    xgb_pipeline = Pipeline([
        ("Data Transformations", ct),
        ("XGBoost Classifier", xgb_classifier)
    ])
    
    nn_pipeline = Pipeline([
        ("Data Transformations", ct),
        ("Neural Network Classifier", nn_classifier)
    ])
    
    # Fit and evaluate each pipeline
    for pipeline, name in [(rfc_pipeline, 'Random Forest'), 
                           (logistic_pipeline, 'Logistic Regression'), 
                           (svm_rbf_pipeline, 'SVM with RBF kernel'),
                           (xgb_pipeline, 'XGBoost Classifier'),
                           (nn_pipeline, 'Neural Network Classifier')]:
        pipeline.fit(X_train, y_train)
        train_accuracy = pipeline.score(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)
        print(f"{name} Training Accuracy: {test_accuracy:.4f}")
        
        # Save the model
        model_save_path = os.path.join(args.model_dir, f"{name.lower().replace(' ', '_')}_model.joblib")
        joblib.dump(pipeline, model_save_path)
        print(f"Model Saved At: {model_save_path}")

if __name__ == "__main__":
    main()
