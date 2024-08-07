import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_outliers(df):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    abs_z_scores = np.abs(scaled_df)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return df[filtered_entries]
