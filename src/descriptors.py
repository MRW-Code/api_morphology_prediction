try:
    from rdkit import Chem
    from mordred import Calculator, descriptors
except:
    print('Descriptor packages must be loaded, imports not working')
    pass

import pandas as pd
from src.utils import args

def descriptors_from_smiles(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(p) for p in smiles_list]
    mols_updated = [mol for mol in mols if isinstance(mol, Chem.Mol)]
    return pd.DataFrame(calc.pandas(mols_updated, nproc=4))

def clean_desciptors(desc_df):
    return desc_df.dropna(axis=1).select_dtypes(exclude=['object'])

def get_desc_set(smiles_list):
    if args.load_data:
        desc_df_clean = pd.read_csv('./data/descriptors_df.csv', index_col=0)
        print(f'Descriptors loaded from data folder, size = {desc_df_clean.shape}')
    else:
        desc_df = descriptors_from_smiles(smiles_list)
        desc_df.index = smiles_list
        desc_df_clean = clean_desciptors(desc_df)
        desc_df_clean.to_csv('./data/descriptors_df.csv')
        print(f'Descriptors calc and saved to data folder, size = {desc_df_clean.shape}')
    return desc_df_clean