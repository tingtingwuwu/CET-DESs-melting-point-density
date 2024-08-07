import csv
import requests
import concurrent.futures
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/CSV'
cache = {}

def get_smiles(name):
    if name in cache:
        return cache[name]
    try:
        response = requests.get(base_url.format(name))
        if response.status_code == 200:
            smiles = response.text.split('\n')[1].split(',')[1]
            cache[name] = smiles
            return smiles
    except Exception as e:
        print(f"Error fetching SMILES for {name}: {e}")
    return 'Not Found'

def fetch_smiles(names):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        smiles = list(executor.map(get_smiles, names))
    return smiles

def preprocess_data(df):
    df = df.dropna()
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    abs_z_scores = np.abs(scaled_df)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return df[filtered_entries]
