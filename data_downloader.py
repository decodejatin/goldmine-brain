import os
import urllib.request
import zipfile

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/PAXGUSDT/1s/"

def download_month(year: int, month: int):
    month_str = f"{month:02d}"
    filename = f"PAXGUSDT-1s-{year}-{month_str}.zip"
    url = f"{BASE_URL}{filename}"
    zip_path = os.path.join(RAW_DATA_DIR, filename)
    
    csv_filename = filename.replace('.zip', '.csv')
    if os.path.exists(os.path.join(RAW_DATA_DIR, csv_filename)):
        print(f"[+] Data for {year}-{month_str} already exists. Skipping download.")
        return

    print(f"[*] Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"[+] Downloaded {filename}")
        
        print(f"[*] Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
            extracted_files = zip_ref.namelist()
            for ef in extracted_files:
                print(f"  -> Extracted: {ef}")
        
        os.remove(zip_path)
        print(f"[+] Cleaned up {filename}")
        
    except Exception as e:
        print(f"[-] Failed to download or extract {filename}: {e}")

if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    # Testing with 1 month (January 2024)
    download_month(2024, 1)
