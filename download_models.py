#!/usr/bin/env python3
"""
Download PAT Model Weights for Big Mood Detector

This script downloads the pre-trained Pretrained Actigraphy Transformer (PAT) model weights
from the official Dropbox links.
"""

import os
import urllib.request
import sys
from pathlib import Path

# PAT Model URLs (29k NHANES dataset - 2003-2014)
MODELS = {
    "PAT-L_29k_weights.h5": "https://www.dropbox.com/scl/fi/exk40hu1nxc1zr1prqrtp/PAT-L_29k_weights.h5?rlkey=t1e5h54oob0e1k4frqzjt1kmz&st=7a20pcox&dl=1",
    "PAT-M_29k_weights.h5": "https://www.dropbox.com/scl/fi/hlfbni5bzsfq0pynarjcn/PAT-M_29k_weights.h5?rlkey=frbkjtbgliy9vq2kvzkquruvg&st=mxc4uet9&dl=1", 
    "PAT-S_29k_weights.h5": "https://www.dropbox.com/scl/fi/12ip8owx1psc4o7b2uqff/PAT-S_29k_weights.h5?rlkey=ffaf1z45a74cbxrl7c9i2b32h&st=mfk6f0y5&dl=1"
}

def download_file(url, filename, target_dir):
    """Download a file with progress indication"""
    filepath = target_dir / filename
    
    if filepath.exists():
        print(f"✅ {filename} already exists, skipping...")
        return True
    
    print(f"📥 Downloading {filename}...")
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                sys.stdout.write(f"\r   Progress: {percent}% ({downloaded // 1024 // 1024}MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✅ {filename} downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        return False

def main():
    print("🧠 Big Mood Detector - PAT Model Weights Downloader")
    print("=" * 60)
    
    # Create target directory
    target_dir = Path("reference_repos/Pretrained-Actigraphy-Transformer/model_weights")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Download directory: {target_dir}")
    print(f"🎯 Models to download: {len(MODELS)}")
    print()
    
    success_count = 0
    for filename, url in MODELS.items():
        if download_file(url, filename, target_dir):
            success_count += 1
        print()
    
    print("=" * 60)
    if success_count == len(MODELS):
        print("🎉 All model weights downloaded successfully!")
        print()
        print("📋 Available models:")
        for filename in MODELS.keys():
            filepath = target_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"   • {filename} ({size_mb:.1f}MB)")
        print()
        print("🚀 Ready to use! Check the Fine-tuning notebooks in:")
        print("   reference_repos/Pretrained-Actigraphy-Transformer/Fine-tuning/")
    else:
        print(f"⚠️  Downloaded {success_count}/{len(MODELS)} models successfully")
        print("   Some downloads may have failed. Try running the script again.")

if __name__ == "__main__":
    main() 