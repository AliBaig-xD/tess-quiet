import pandas as pd
from astroquery.mast import Catalogs

ANOMALIES_FILE = 'results/anomalies.parquet'
OUT_FILE       = 'results/anomalies_with_tic.parquet'

TIC_COLS = ['ID', 'Teff', 'rad', 'mass', 'lum', 'Tmag', 'ra', 'dec', 'logg', 'MH']

def fetch_tic_batch(tic_ids, batch_size=1000):
    all_results = []
    for i in range(0, len(tic_ids), batch_size):
        batch = tic_ids[i:i + batch_size]
        try:
            result = Catalogs.query_criteria(catalog='TIC', ID=batch)
            if result is not None and len(result) > 0:
                all_results.append(result[TIC_COLS].to_pandas())
        except Exception as e:
            print(f"  Batch {i//batch_size + 1} failed: {e}")
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame(columns=TIC_COLS)

def main():
    anomalies = pd.read_parquet(ANOMALIES_FILE)
    tic_ids   = anomalies['tic_id'].astype(int).tolist()
    print(f"Fetching TIC parameters for {len(tic_ids)} stars...")

    tic_df = fetch_tic_batch(tic_ids)
    tic_df['tic_id'] = tic_df['ID'].astype(str)

    merged = anomalies.merge(tic_df, on='tic_id', how='left')
    merged.to_parquet(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")
    print(f"Stars with Teff: {merged['Teff'].notna().sum()}")

if __name__ == '__main__':
    main()