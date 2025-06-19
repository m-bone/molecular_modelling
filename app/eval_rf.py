import pickle
import pandas as pd

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

def evaluate_random_forest(smiles: str) -> int:
    """
    Evaluate a Random Forest model on a given SMILES string.

    :param smiles: SMILES representation of the molecule.
    :return: Predicted class label (0 or 1).
    """
    # Convert SMILES to RDKit Mol object
    mol = Chem.MolFromSmiles(smiles) # type: ignore
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # Generate molecular descriptors
    desc_df = pd.DataFrame([Descriptors.CalcMolDescriptors(mol)])

    with open("rf_columns_dipole_moment_3.0.pkl", "rb") as f:
        columns = pickle.load(f)

    # Prepare input for the model - NOTE: Check order is correct
    desc_df = desc_df[columns]

    # Load the pre-trained Random Forest model
    with open('rf_model_dipole_moment_3.0.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    # Predict using the Random Forest model
    # Adding .values suppresses odd warning about being trained with column names
    prediction = rf_model.predict(desc_df.values)

    return int(prediction[0])  # Return as integer (0 or 1)

if __name__ == "__main__":
    # Example usage
    import os
    os.chdir("/home/mab-desk/Documents/molecular_modelling/app/") # Shouldn't be needed in future
    
    smiles_example = "CCO"  # Ethanol
    
    result = evaluate_random_forest(smiles_example)
    print(f"Predicted class for {smiles_example}: {result}")