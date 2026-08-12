import os
import json
import subprocess
import time
import sys

# Load .env variables to authenticate Kaggle API
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key] = val.strip('"')
    
    # Kaggle's new CLI uses KAGGLE_API_TOKEN
    if "KAGGLE_KEY" in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_KEY"]

def generate_metadata(username="jatinjalandhra", slug="goldmine-ppo-training"):
    metadata = {
      "id": f"{username}/{slug}",
      "title": "Goldmine PPO Training",
      "code_file": "rl_agent.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_internet": "true",
      "dataset_sources": [],
      "competition_sources": [],
      "kernel_sources": []
    }
    with open(os.path.join(os.path.dirname(__file__), "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] Generated kernel-metadata.json for {username}/{slug}")

def push_to_kaggle():
    print("[*] Pushing to Kaggle via 'kaggle kernels push'...")
    # Requires Kaggle API token in ~/.kaggle/kaggle.json
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", os.path.dirname(__file__)], check=True)

def monitor_and_download(username="jatinjalandhra", slug="goldmine-ppo-training"):
    print("[*] Monitoring Kaggle Job Status...")
    kernel_id = f"{username}/{slug}"
    while True:
        result = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", kernel_id], 
                                capture_output=True, text=True)
        status = result.stdout.lower()
        print(f"Status: {status.strip()}")
        
        if "complete" in status:
            print("[+] Job completed! Downloading output...")
            out_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(out_dir, exist_ok=True)
            subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", kernel_id, "-p", out_dir], check=True)
            print(f"[+] Downloaded outputs to {out_dir}/")
            break
        elif "error" in status or "failed" in status:
            print("[-] Kaggle job failed.")
            break
            
        time.sleep(30)

if __name__ == "__main__":
    generate_metadata()
    # Activate actual Kaggle push for the live demo
    push_to_kaggle()
    monitor_and_download()
    print("[+] Kaggle headless launcher successfully configured.")
