"""
example_standalone_analysis.py

This script demonstrates how to load a surface (LAZ/LAS/CSV/TXT), preprocess it, 
and extract all texture parameters (including 3D areal stats and Core Slope) 
without using the Streamlit GUI. 

Usage:
  python example_standalone_analysis.py path/to/your/file.laz
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Import the TextureLab engine modules
# Adjust the sys.path if you move this file so it can find TextureLab/src
sys.path.insert(0, str(Path(__file__).resolve().parent / "TextureLab"))

from src.data_io import load_surface
from src.preprocessing import preprocess_surface, PreprocessConfig
from src.descriptors import compute_all, aggregate_profiles

def analyze_file(filepath, dx=1.0, dy=1.0, units_xy="mm", units_z="mm"):
    print(f"📄 Loading {filepath}...")
    
    # 1. Load the surface grid
    grid = load_surface(filepath, dx=dx, dy=dy, units_xy=units_xy, units_z=units_z)
    print(f"  Grid size: {grid.ny}×{grid.nx}")
    
    # 2. Configure preprocessing
    cfg = PreprocessConfig(
        plane_removal="plane",  # options: "none", "mean", "plane", "poly2"
        outlier_method="hampel", 
        outlier_win=7,
        outlier_sig=3.0,
        fill_gaps=True,
        filter_type="none"      # options: "none", "fft_bandpass", "iir_bandpass"
    )
    
    # 3. Preprocess and extract profiles
    # direction: "longitudinal" or "transverse"
    # every_n: extract every Nth profile (1 means extract all profiles)
    print("🧹 Preprocessing and extracting profiles...")
    z_proc, profiles, warnings = preprocess_surface(
        grid.z, dx=dx, dy=dy, cfg=cfg, direction="longitudinal", every_n=1
    )
    
    for w in warnings:
        print(f"  ⚠ {w}")
    print(f"  ✅ {len(profiles)} profiles extracted")
    
    # 4. Compute all parameters
    print("⚙️ Computing descriptors (Profile & Surface)...")
    per_profile_results, areal_results = compute_all(profiles, z_proc, dx=dx, dy=dy)
    
    # 5. Aggregate the profile results (mean, median, etc.)
    aggregated = aggregate_profiles(per_profile_results, mode="mean")
    
    # 6. Extract the Abbott-Firestone parameters and calculate Core Slope
    # Profile Slope
    rk = aggregated.get("Rk", 0)
    mr1 = aggregated.get("Mr1", 0)
    mr2 = aggregated.get("Mr2", 0)
    profile_slope = (rk / (mr2 - mr1) * 100) if (mr2 - mr1) > 0 else 0
    aggregated["Slope (%)"] = profile_slope
    
    # Surface Slope
    sk = areal_results.get("Sk", 0)
    smr1 = areal_results.get("SMr1", 0)
    smr2 = areal_results.get("SMr2", 0)
    surface_slope = (sk / (smr2 - smr1) * 100) if (smr2 - smr1) > 0 else 0
    areal_results["Slope (%)"] = surface_slope
    
    # 7. Print Summary
    print("\n" + "="*50)
    print("📊 RESULTS SUMMARY")
    print("="*50)
    print("\n--- Profile Parameters (Averaged) ---")
    for k, v in aggregated.items():
        if not k.endswith(('_std', '_P10', '_P90')):
            print(f"  {k:15s}: {v:.4f}")
            
    print("\n--- 3D Surface Parameters (Areal) ---")
    for k, v in areal_results.items():
        print(f"  {k:15s}: {v:.4f}")
        
    print("="*50)
    
    return aggregated, areal_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone TextureLab Analysis")
    parser.add_argument("file", type=str, help="Path to the surface file (LAZ/CSV/TXT)")
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing X (mm)")
    parser.add_argument("--dy", type=float, default=1.0, help="Grid spacing Y (mm)")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)
        
    analyze_file(args.file, dx=args.dx, dy=args.dy)
