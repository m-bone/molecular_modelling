import pickle
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


seed = 2
np.random.seed(seed)

def train_random_forest(
        prop_target: str, 
        target_value: float, 
        desc_df: pd.DataFrame, 
        targets_df: pd.DataFrame,
        balance_data: bool = False):
    """
    Train a Random Forest Classifier on the provided molecular descriptors and target values.
    Uses Imblearns BalancedRandomForestClassifier for handling class imbalance instead of under-sampling the dataset.


    :param prop_target: The property to predict (e.g., "Dipole Moment").
    :param target_value: The threshold value for the target property.
    :param desc_df: DataFrame containing molecular descriptors.
    :param targets_df: DataFrame containing target values.
    :param balance_data: Whether to balance the dataset using Random Under Sampling.
    """
    
    # Ensure the target property exists in the targets DataFrame
    if prop_target not in targets_df.columns:
        raise ValueError(f"Target property '{prop_target}' not found in targets DataFrame.")
    
    # Find indices of rows with missing data in desc_df
    missing_indices = desc_df[desc_df.isnull().any(axis=1)].index

    # Drop rows with missing data from desc_df and get the indices of the remaining rows
    desc_df.dropna(inplace=True)
    targets_df = targets_df.drop(missing_indices)

    # Remove columns with no variance
    desc_df = desc_df.loc[:, desc_df.var() != 0]

    # Sort columns by variance and remove gdb_tag
    desc_df = desc_df[desc_df.var().sort_values(ascending=False).index]
    desc_df = desc_df.drop(columns=['gdb_tag'], errors='ignore')

    # Select first 60 columns
    desc_df = desc_df.iloc[:, :60]

    # Select target values
    targets = np.where(targets_df[prop_target] > target_value, 1, 0)
    print(f"Target split: {np.sum(targets)} positive, {len(targets) - np.sum(targets)} negative")
    print(f"Percentage of positive targets: {np.mean(targets) * 100:.2f}%")

    # Balance data using Imblearn to under-sample
    if balance_data:
        rus = RandomUnderSampler(random_state=seed)
        desc_df, targets = rus.fit_resample(desc_df, targets) # type: ignore

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(desc_df, targets, test_size=0.2, random_state=seed)
    print("Training set size: ", len(X_train))

    clf = BalancedRandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Predicting {prop_target} at {target_value}...")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the model to a file
    model_filename = f"rf_model_{prop_target}_{target_value}.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)

    # Output the column names of the descriptors used
    # Needed as the model is trained on columns sorted by variance
    with open(f"rf_columns_{prop_target}_{target_value}.pkl", 'wb') as f:
        pickle.dump(desc_df.columns.tolist(), f)

if __name__ == "__main__":
    from utils import create_connection

    # Load data from PostgreSQL database
    engine = create_connection()
    desc_df = pd.read_sql_table('gdb9_descriptors', engine)
    prop_df = pd.read_sql_table('gdb9_properties', engine)

    # Clip DFs for testing
    desc_df = desc_df.iloc[:10000, :]
    prop_df = prop_df.iloc[:10000, :]

    prop_target = "dipole_moment"
    target_value = 3.0

    balance_data = False
    train_random_forest(prop_target, target_value, desc_df, prop_df, balance_data)