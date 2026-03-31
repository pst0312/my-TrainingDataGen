"""
batch_generate.py
=================
User-facing batch generation interface for synthetic spectroscopy data.

Features:
  - Interactive prompts for visual and data complexity ranges
  - Automatic batch folder creation with timestamp
  - Routes all outputs (PNG, CSV) to organized batch directory
  - Validates input and provides defaults
"""

import subprocess
import sys
import os
import glob
from datetime import datetime


def parse_complexity_range(user_input, default_min=1, default_max=10):
    """
    Parse user input for complexity range in "min,max" format.
    If empty, return defaults. If invalid, return None.
    """
    if not user_input or user_input.strip() == "":
        return (default_min, default_max)
    
    try:
        parts = user_input.strip().split(",")
        if len(parts) == 2:
            min_val = int(parts[0].strip())
            max_val = int(parts[1].strip())
            # Validate ranges
            min_val = max(1, min(min_val, 10))
            max_val = max(1, min(max_val, 10))
            return (min(min_val, max_val), max(min_val, max_val))
        else:
            print(f"Invalid format. Expected 'min,max' (e.g., '2,8'), got '{user_input}'")
            return None
    except ValueError:
        print(f"Invalid input. Please use numbers: 'min,max' (e.g., '1,5')")
        return None


def main():
    try:
        n = int(input("How many spectra would you like to generate? "))
    except Exception:
        print("Invalid input. Please enter an integer.")
        sys.exit(1)
    if n < 1:
        print("Please enter a positive integer.")
        sys.exit(1)
    
    # Get visual complexity range
    while True:
        print("\nEnter Visual Complexity range (1-10) [Press Enter for default 1,10]:")
        vis_input = input("→ ").strip()
        vis_range = parse_complexity_range(vis_input, 1, 10)
        if vis_range is not None:
            min_vis, max_vis = vis_range
            break
    
    # Get data complexity range
    while True:
        print("\nEnter Data Complexity range (1-10) [Press Enter for default 1,10]:")
        data_input = input("→ ").strip()
        data_range = parse_complexity_range(data_input, 1, 10)
        if data_range is not None:
            min_data, max_data = data_range
            break
    
    # Calculate averages
    avg_vis = (min_vis + max_vis) / 2.0
    avg_data = (min_data + max_data) / 2.0
    
    # Create batch folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = f"batch_vis{avg_vis:.1f}_data{avg_data:.1f}_{timestamp}"
    
    try:
        os.makedirs(batch_folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating batch folder '{batch_folder}': {e}")
        sys.exit(1)
    
    print(f"\nConfiguration Summary:")
    print(f"  Visual Complexity: {min_vis}-{max_vis} (avg: {avg_vis:.1f})")
    print(f"  Data Complexity: {min_data}-{max_data} (avg: {avg_data:.1f})")
    print(f"  Spectra to generate: {n}")
    print(f"  Output folder: {batch_folder}/")
    
    csv_files = []
    for i in range(1, n + 1):
        print(f"\n--- Generating spectrum {i} of {n} ---")
        try:
            # Build command with complexity arguments
            cmd = [
                sys.executable,
                "spectrum_generator.py",
                str(i),
                "--min-vis", str(min_vis),
                "--max-vis", str(max_vis),
                "--min-data", str(min_data),
                "--max-data", str(max_data),
            ]
            subprocess.run(cmd, check=True)
            
            # Move generated files to batch folder
            # Pattern: spectrum_*_multiline_{i}.*
            png_matches = glob.glob(f"spectrum_*_multiline_{i}.png")
            csv_matches = glob.glob(f"spectrum_data_*_multiline_{i}.csv")
            
            for file in png_matches + csv_matches:
                try:
                    new_path = os.path.join(batch_folder, file)
                    os.rename(file, new_path)
                    if file.endswith('.csv'):
                        csv_files.append(new_path)
                except OSError as e:
                    print(f"Warning: Could not move {file}: {e}")
                    
        except subprocess.CalledProcessError as e:
            print(f"Error running spectrum_generator.py: {e}")
            break
    
    # Save the list of CSV files for validation
    csv_list_path = os.path.join(batch_folder, "generated_csv_files.txt")
    with open(csv_list_path, "w") as f:
        for csvf in csv_files:
            f.write(csvf + "\n")
    
    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"CSV files generated: {len(csv_files)}/{n}")
    print(f"All outputs saved to: {batch_folder}/")
    print(f"CSV file list: {csv_list_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
