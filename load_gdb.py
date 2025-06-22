# Python 3 - MAB 2025
import os
import natsort
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from utils import create_connection

# Header names from paper here: https://www.nature.com/articles/sdata201422/tables/4
GDB_HEADER = ["gdb_tag", "rotational_constant_a", "rotational_constant_b", "rotational_constant_c", 
              "dipole_moment", "isotropic_polarizability", "energy_homo", "energy_lumo", "homo_lumo_gap",
              "electronic_spatial_extent", "vibrational_energy", "internal_energy_az", "internal_energy_rt",
              "enthalpy", "free_energy", "heat_capacity"]

def get_gdb_properties(file_dir: str, file_name: str, gdb_header:list) -> dict:
    """
    Read gdb file and create dict combining header properties with the pre-defined header strings
    """
    with open(file_dir + "/" + file_name, "r") as f:
        lines = f.readlines()
        prop_line = lines[1].split()
        prop = {key: float(val) for key, val in zip(gdb_header, prop_line[1:])} # indexing skips "GDB" first element
    
    return prop

def get_smiles_inchi(file_dir: str, file_name: str) -> tuple[list, list]:
    """
    Find and return the SMILES and InChI strings for each molecule.
    """
    with open(file_dir + "/" + file_name, "r") as f:
        lines = f.readlines()
        smiles = lines[-2].split()
        inchi = lines[-1].split()
        
    return smiles, inchi

def build_gdb_df(gdb_dir: str, gdb_files: list[str], keep_files) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two dataframes of GDB properties and molecular descriptors from RDKit.
    Save dataframes to local PostgreSQL database.

    Two choices for handling difficult files:
    Either convert all to SMILES 1 - this is GDB17 file which dictates the start of QM calc
    Dump all difficult files for fear that the calculated QM values are garbage
    """
    gdb_values = []
    descriptions = []
    smiles_list = []
    inchi_list = []
    error_file = []
    persist_error = []
    for gdb_file in gdb_files:
        smiles, inchi = get_smiles_inchi(gdb_dir, gdb_file)

        # Select the first SMILES and second InChI
        smiles = smiles[0]
        inchi = inchi[1]

        try:
            molecule = Chem.MolFromInchi(inchi, sanitize=False, treatWarningAsError=True)
            Chem.SanitizeMol(molecule)
        except (Chem.AtomValenceException, Chem.InchiReadWriteError) as e:
            error_file.append(gdb_file)
            if keep_files:
                molecule = Chem.MolFromSmiles(smiles)
            else:
                continue

        if molecule is None:
            persist_error.append(gdb_file)
    
        desc = pd.DataFrame([Descriptors.CalcMolDescriptors(molecule)])
        gdb_values.append(get_gdb_properties(gdb_dir, gdb_file, GDB_HEADER))
        descriptions.append(desc)
        smiles_list.append(smiles)
        inchi_list.append(inchi)

    # Create GDB dataframe with molecule strings
    prop_df = pd.DataFrame(gdb_values)
    # Prepend SMILES and InChI columns
    prop_df.insert(0, "smiles", smiles_list)
    prop_df.insert(1, "inchi", inchi_list)

    # Create KDKit descriptors dataframe
    desc_df = pd.concat(descriptions, ignore_index=True)
    desc_df.insert(0, "gdb_tag", prop_df["gdb_tag"])

    print(f"Persist error files:\n")
    for name in persist_error:
        print(f"{name}\n")

    return prop_df, desc_df


if __name__ == "__main__":
    gdb_dir = "/home/mab-desk/Documents/atom_dbs/gdb9"
    gdb_files = natsort.natsorted(os.listdir(gdb_dir))
    gdb_idx = (0, 133885)
    keep_files = False

    prop_df, desc_df = build_gdb_df(gdb_dir, gdb_files[gdb_idx[0]:gdb_idx[1]], keep_files)
    engine = create_connection()
    print(f"Length of DB: {len(desc_df)}")
    prop_df.to_sql('gdb9_properties_cut', engine, if_exists='replace', index=False)
    desc_df.to_sql('gdb9_descriptors_cut', engine, if_exists='replace', index=False)
    print(f"Saved GDB properties and descriptors to PostgreSQL database.")