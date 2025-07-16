#!/usr/bin/env python3
"""
Extract Learned Positional Embeddings from PAT Repository

This script extracts learned positional embeddings from the original
PAT repository models if available, for potential fine-tuning use.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def extract_learned_embeddings(weights_path: Path, output_dir: Path):
    """Extract learned positional embeddings if present in the weights."""
    print(f"\n{'='*70}")
    print(f"Extracting embeddings from: {weights_path.name}")
    print(f"{'='*70}")
    
    try:
        with h5py.File(weights_path, "r") as f:
            # Common names for positional embeddings in transformer models
            embedding_names = [
                "pos_embedding",
                "positional_embedding",
                "position_embeddings",
                "pos_embed",
                "learned_pos_embed",
                "cls_positional_encoding",
                "positional_encoding"
            ]
            
            found_embeddings = {}
            
            # Search for positional embeddings
            def search_group(group, path=""):
                for key in group.keys():
                    item = group[key]
                    full_path = f"{path}/{key}" if path else key
                    
                    if isinstance(item, h5py.Dataset):
                        # Check if this might be positional embeddings
                        for emb_name in embedding_names:
                            if emb_name in key.lower():
                                shape = item.shape
                                print(f"✅ Found potential embedding: {full_path}")
                                print(f"   Shape: {shape}")
                                found_embeddings[full_path] = np.array(item)
                    elif isinstance(item, h5py.Group):
                        search_group(item, full_path)
            
            search_group(f)
            
            if found_embeddings:
                # Save the embeddings
                output_file = output_dir / f"{weights_path.stem}_embeddings.npz"
                np.savez_compressed(output_file, **found_embeddings)
                print(f"\n✅ Saved {len(found_embeddings)} embeddings to: {output_file}")
                
                # Show embedding info
                for name, emb in found_embeddings.items():
                    print(f"\n📊 {name}:")
                    print(f"   Shape: {emb.shape}")
                    print(f"   Min: {emb.min():.4f}")
                    print(f"   Max: {emb.max():.4f}")
                    print(f"   Mean: {emb.mean():.4f}")
                    print(f"   Std: {emb.std():.4f}")
                
                return True
            else:
                print("\n⚠️  No learned positional embeddings found")
                print("   The model may use sinusoidal embeddings only")
                return False
                
    except Exception as e:
        print(f"❌ Error extracting embeddings: {e}")
        return False


def main():
    """Extract learned embeddings from all available PAT models."""
    weights_dir = Path("model_weights/pat/pretrained")
    output_dir = Path("model_weights/pat/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also check if we have the original PAT repo
    pat_repo = Path("reference_repos/Pretrained-Actigraphy-Transformer")
    
    print("LEARNED POSITIONAL EMBEDDINGS EXTRACTION")
    print("="*70)
    
    # Check PAT repo
    if pat_repo.exists():
        print(f"✅ Found PAT repository at: {pat_repo}")
        
        # Look for model files in the repo
        repo_models = list(pat_repo.rglob("*.h5"))
        if repo_models:
            print(f"\n📁 Found {len(repo_models)} H5 files in PAT repo:")
            for model_path in repo_models[:5]:  # Show first 5
                print(f"   - {model_path.relative_to(pat_repo)}")
            
            # Try to extract from repo models
            for model_path in repo_models:
                if "checkpoint" in str(model_path) or "weights" in str(model_path):
                    extract_learned_embeddings(model_path, output_dir)
    else:
        print(f"❌ PAT repository not found at: {pat_repo}")
        print("   Clone it with: git clone https://github.com/njmei/Pretrained-Actigraphy-Transformer")
    
    # Check our pretrained weights
    print(f"\n📁 Checking our pretrained weights...")
    models = [
        "PAT-S_29k_weights.h5",
        "PAT-M_29k_weights.h5", 
        "PAT-L_29k_weights.h5",
    ]
    
    found_count = 0
    for filename in models:
        weights_path = weights_dir / filename
        if weights_path.exists():
            if extract_learned_embeddings(weights_path, output_dir):
                found_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*70}")
    
    if found_count > 0:
        print(f"✅ Extracted embeddings from {found_count} models")
        print("\nTo use learned embeddings:")
        print("1. Load the .npz file with the embeddings")
        print("2. Replace sinusoidal embeddings in DirectPATModel")
        print("3. Fine-tune if needed for your specific domain")
    else:
        print("ℹ️  No learned embeddings found in the weights")
        print("   The PAT models appear to use sinusoidal embeddings")
        print("   This is actually preferable for generalization!")
        print("\n   Sinusoidal embeddings:")
        print("   - Don't require training")
        print("   - Generalize to any sequence length")
        print("   - Work well for periodic signals like circadian rhythms")


if __name__ == "__main__":
    main()