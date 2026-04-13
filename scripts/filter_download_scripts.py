"""
Phase C — Filter Download Scripts
Filters the bulk MAST curl scripts to only include boring (non-variable) stars.
Output: scripts/filtered/filtered_sector_N_lc.sh for N in 1..5
"""

import os
import re

BORING_IDS_FILE = 'data/boring_tic_ids.txt'
SCRIPT_DIR      = 'scripts'
OUT_DIR         = 'scripts/filtered'

os.makedirs(OUT_DIR, exist_ok=True)


def extract_tic_from_curl_line(line):
    """
    Extract TIC ID from a MAST curl line.
    """
    match = re.search(r'-s\d+-(\d+)-', line)
    if match:
        return match.group(1)
    return None


def main():
    with open(BORING_IDS_FILE) as f:
        boring_ids = set(line.strip() for line in f if line.strip())

    for sector_num in range(1, 6):
        script_path = os.path.join(SCRIPT_DIR, f'tesscurl_sector_{sector_num}_lc.sh')
        out_path    = os.path.join(OUT_DIR, f'filtered_sector_{sector_num}_lc.sh')

        if not os.path.exists(script_path):
            print(f"WARNING: {script_path} not found — skipping sector {sector_num}")
            continue

        kept  = 0
        total = 0
        with open(script_path) as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                if not line.startswith('curl'):
                    f_out.write(line)
                    continue
                total += 1
                tic_id = extract_tic_from_curl_line(line)
                if tic_id and tic_id in boring_ids:
                    f_out.write(line)
                    kept += 1

        print(f"Sector {sector_num}: kept {kept}/{total} curl lines → {out_path}")


if __name__ == '__main__':
    main()
