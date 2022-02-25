try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except:
    print('Images packages must be loaded, imports not working')
    pass
import pandas as pd

def images_from_dataset(df):
    for count, api in enumerate(df.api):
        smiles = df.SMILES[count]
        label = df.eye_morphology[count]
        filepath = f'./data/images/{label}/{api}.png'
        mol = Chem.MolFromSmiles(smiles)
        draw = Draw.MolToFile(mol, filepath, size=[250,250])
    return None
