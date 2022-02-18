import pandas as pd
from src.utils import args
from src.descriptors import get_desc_set

def get_lab_data():
    if args.user == 'matt':
        path = './data/summer_hts_data_matt.csv'
    elif args.user == 'laura':
        path = './data/summer_hts_data.csv'
    else:
        raise AttributeError('dataset does not exist')
    lab_df = pd.read_csv(path, index_col=0).reset_index().drop('index', axis=1)
    return lab_df

def get_ml_df():
    lab_data = get_lab_data()
    labels = lab_data.loc[:, ['SMILES', 'eye_morphology']]
    desc = get_desc_set(lab_data.loc[:, 'SMILES'])
    df = pd.merge(labels, desc, left_on='SMILES', right_index=True)
    return df


