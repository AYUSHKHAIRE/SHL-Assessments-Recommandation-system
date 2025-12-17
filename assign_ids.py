import pandas as pd
from uuid import uuid4

def assign_dataset_ids(filepath: str) -> list:
    """
    Assign unique dataset IDs to each entry in the CSV file.
    """
    df = pd.read_csv(filepath)
    if 'dataset_id' not in df.columns:
        df['dataset_id'] = [str(uuid4()) for _ in range(len(df))]
    else:
        # Assign new IDs only to rows without an ID
        df['dataset_id'] = df['dataset_id'].apply(
            lambda x: str(uuid4()) if pd.isna(x) or x == '' else x
        )
    df.to_csv(filepath, index=False)
    return df['dataset_id'].tolist()

assign_dataset_ids('data/assessments_details.csv')