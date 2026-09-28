import os
import argparse
import gzip
import shutil
import pandas as pd
import requests

HOSP_TABLES = [
    "admissions.csv.gz",
    "diagnoses_icd.csv.gz",
    "drgcodes.csv.gz",
    "emar.csv.gz",
    "emar_detail.csv.gz",
    "hcpcsevents.csv.gz",
    "labevents.csv.gz",
    "microbiologyevents.csv.gz",
    "omr.csv.gz",
    "patients.csv.gz",
    "pharmacy.csv.gz",
    "poe.csv.gz",
    "poe_detail.csv.gz",
    "prescriptions.csv.gz",
    "procedures_icd.csv.gz",
    "services.csv.gz",
    "transfers.csv.gz"
]

ICU_TABLES = [
    "chartevents.csv.gz",
    "datetimeevents.csv.gz",
    "icustays.csv.gz",
    "ingredientevents.csv.gz",
    "inputevents.csv.gz",
    "outputevents.csv.gz",
    "procedureevents.csv.gz"
]

DICT_TABLES = [
    "hosp/d_hcpcs.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/d_icd_procedures.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/provider.csv.gz",
    "icu/d_items.csv.gz",
    "icu/caregiver.csv.gz"
]


def download_unzip(url, auth, out_path_csv, out_path_gz):
    print(f"Downloading full dictionary table: {url}")
    headers = {"User-Agent": "Wget/1.21.2"}
    resp = requests.get(url, auth=auth, stream=True, headers=headers)
    resp.raise_for_status()
    # Write to .csv.gz
    with open(out_path_gz, 'wb') as f_gz:
        shutil.copyfileobj(resp.raw, f_gz)
    
    # Extract to .csv
    with gzip.open(out_path_gz, 'rb') as f_in, open(out_path_csv, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def get_pneumonia_subjects(base_url, auth, limit=None):
    print("Fetching d_icd_diagnoses to find pneumonia ICD codes...")
    d_icd_url = f"{base_url}hosp/d_icd_diagnoses.csv.gz"
    headers = {"User-Agent": "Wget/1.21.2"}
    resp = requests.get(d_icd_url, auth=auth, stream=True, headers=headers)
    resp.raise_for_status()
    with gzip.open(resp.raw, 'rt') as f:
        d_icd = pd.read_csv(f)
    
    pneumonia_icds = d_icd[d_icd['long_title'].str.contains('pneumonia', case=False, na=False)]
    icd_codes = set(pneumonia_icds['icd_code'].astype(str))
    icd_versions = set(pneumonia_icds['icd_version'].astype(str))
    
    print(f"Found {len(icd_codes)} pneumonia-related ICD codes.")
    
    print("Fetching diagnoses_icd to find subjects with pneumonia...")
    diag_url = f"{base_url}hosp/diagnoses_icd.csv.gz"
    resp = requests.get(diag_url, auth=auth, stream=True, headers=headers)
    resp.raise_for_status()
    
    subjects = set()
    with gzip.open(resp.raw, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=100000, dtype={'icd_code': str, 'icd_version': str}):
            mask = chunk['icd_code'].isin(icd_codes) & chunk['icd_version'].isin(icd_versions)
            subjects.update(chunk.loc[mask, 'subject_id'].unique())
            
            if limit and len(subjects) >= limit:
                break
                
    if limit:
        subjects = set(list(subjects)[:limit])
        
    print(f"Identified {len(subjects)} subjects with pneumonia.")
    return subjects

def process_table(table_path, base_url, auth, subjects, output_dir):
    url = f"{base_url}{table_path}"
    out_path_gz = os.path.join(output_dir, table_path)
    out_path_csv = out_path_gz.replace(".gz", "")
    
    os.makedirs(os.path.dirname(out_path_gz), exist_ok=True)
    
    if table_path in DICT_TABLES:
        download_unzip(url, auth, out_path_csv, out_path_gz)
        return
        
    print(f"Processing and filtering {table_path}...")
    headers = {"User-Agent": "Wget/1.21.2"}
    resp = requests.get(url, auth=auth, stream=True, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to fetch {table_path} - Status code {resp.status_code}")
        return
        
    first_chunk = True
    try:
        with gzip.open(resp.raw, 'rt') as in_f:
            for chunk in pd.read_csv(in_f, chunksize=100000, low_memory=False):
                if 'subject_id' in chunk.columns:
                    filtered = chunk[chunk['subject_id'].isin(subjects)]
                    if not filtered.empty:
                        filtered.to_csv(out_path_csv, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
                        first_chunk = False
                else:
                    # In case a table is not recognized as dict but doesn't have subject_id
                    chunk.to_csv(out_path_csv, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
                    first_chunk = False
    except Exception as e:
        print(f"Error processing {table_path}: {e}")
        return
        
    if first_chunk:
        print(f"Warning: No data matched for {table_path}. Writing empty CSV.")
        open(out_path_csv, 'w').close()
        
    print(f"Compressing {table_path}...")
    with open(out_path_csv, 'rb') as f_in, gzip.open(out_path_gz, 'wb', compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)


def main():
    parser = argparse.ArgumentParser(description="Download and filter MIMIC-IV 3.1 for pneumonia patients.")
    parser.add_argument('--username', required=True, help="PhysioNet username")
    parser.add_argument('--password', required=True, help="PhysioNet password")
    parser.add_argument('--output-dir', default='./mimic-iv-3.1-pneumonia-new', help="Output directory")
    parser.add_argument('--limit', type=int, default=100, help="Limit number of subjects (default: 100)")
    args = parser.parse_args()
    
    base_url = "https://physionet.org/files/mimiciv/3.1/"
    auth = (args.username, args.password)
    
    # First, test auth
    print("Testing PhysioNet authentication...")
    headers = {"User-Agent": "Wget/1.21.2"}
    test_resp = requests.head(base_url, auth=auth, headers=headers)
    if test_resp.status_code in (401, 403):
        print("Authentication failed! Please check your credentials and DUA signing on PhysioNet.")
        return
        
    os.makedirs(os.path.join(args.output_dir, "hosp"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "icu"), exist_ok=True)
    
    # Get subjects
    subjects = get_pneumonia_subjects(base_url, auth, limit=args.limit)
    
    if not subjects:
        print("No pneumonia subjects found!")
        return
        
    all_tables = [f"hosp/{t}" for t in HOSP_TABLES] + [f"icu/{t}" for t in ICU_TABLES] + DICT_TABLES
    
    for table in all_tables:
        process_table(table, base_url, auth, subjects, args.output_dir)
        
    print("Download and filtering complete!")

if __name__ == "__main__":
    main()
