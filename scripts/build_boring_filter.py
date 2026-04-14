"""
Phase B — Build the "Boring Stars" Filter
Excludes known variables via TESS-SVC catalog and SIMBAD TAP query.
Output: data/boring_tic_ids.txt
"""

import pandas as pd
from astroquery.simbad import Simbad
from astropy.table import Table

ALL_TIC_IDS_FILE = 'data/all_tic_ids.txt'
TESS_SVC_FILE    = 'data/tess_svc.csv'
OUT_FILE         = 'data/boring_tic_ids.txt'


def load_all_tic_ids():
    with open(ALL_TIC_IDS_FILE) as f:
        # Strip leading zeros to match TESS-SVC integer format
        return set(str(int(line.strip())) for line in f if line.strip())


def load_tess_svc_ids():
    df = pd.read_csv(TESS_SVC_FILE)
    # TESS-SVC uses 'tess_id' as the TIC identifier column
    col = 'tess_id'
    if col not in df.columns:
        # fallback: find any column with 'id' in the name
        candidates = [c for c in df.columns if c.lower().endswith('id')]
        if not candidates:
            raise ValueError(f"Cannot find TIC ID column. Columns: {df.columns.tolist()}")
        col = candidates[0]
        print(f"Warning: 'tess_id' not found, using '{col}'")
    ids = set(df[col].astype(str).str.strip())
    print(f"  Sample TESS-SVC IDs: {list(ids)[:3]}")
    return ids


def query_simbad_variable_ids(tic_ids):
    """Single TAP batch query — returns set of TIC IDs classified as variable in SIMBAD."""
    upload_table = Table({'tic_id': [f'TIC {t}' for t in list(tic_ids)]})
    adql = """
        SELECT upload.tic_id, basic.main_id, basic.otype
        FROM basic
        JOIN TAP_UPLOAD.my_tics AS upload
          ON basic.main_id = upload.tic_id
        WHERE basic.otype = 'V*..'
    """
    try:
        result = Simbad.query_tap(adql, my_tics=upload_table)
        if result is None or len(result) == 0:
            return set()
        matched = set()
        for row in result:
            num = str(row['tic_id']).replace('TIC', '').strip()
            matched.add(num)
        return matched
    except Exception as e:
        print(f"SIMBAD TAP query error: {e}")
        print("Proceeding without SIMBAD filter — relying on TESS-SVC only.")
        return set()


def main():
    print("Loading all TIC IDs...")
    all_ids = load_all_tic_ids()
    print(f"  Total: {len(all_ids)}")

    print("Loading TESS-SVC known variable IDs...")
    svc_ids = load_tess_svc_ids()
    print(f"  Known variables in TESS-SVC: {len(svc_ids)}")

    after_svc = all_ids - svc_ids
    print(f"  After TESS-SVC removal: {len(after_svc)}")

    print("Querying SIMBAD for additional known variables...")
    if len(after_svc) > 80000:
        chunks = [list(after_svc)[i:i + 50000] for i in range(0, len(after_svc), 50000)]
        simbad_vars = set()
        for i, chunk in enumerate(chunks):
            print(f"  SIMBAD batch {i + 1}/{len(chunks)}...")
            simbad_vars |= query_simbad_variable_ids(set(chunk))
    else:
        simbad_vars = query_simbad_variable_ids(after_svc)

    print(f"  Additional known variables in SIMBAD: {len(simbad_vars)}")

    boring_ids = after_svc - simbad_vars
    print(f"  Final boring star count: {len(boring_ids)}")

    with open(OUT_FILE, 'w') as f:
        for tic in sorted(boring_ids):
            f.write(tic + '\n')

    print(f"Saved to {OUT_FILE}")


if __name__ == '__main__':
    main()
