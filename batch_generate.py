import subprocess
import sys


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
    
    print(f"\nConfiguration Summary:")
    print(f"  Visual Complexity: {min_vis}-{max_vis}")
    print(f"  Data Complexity: {min_data}-{max_data}")
    print(f"  Spectra to generate: {n}")
    
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
            
            # Try to find the generated CSV file (pattern: spectrum_data_*_multiline_{i}.csv)
            import glob
            matches = glob.glob(f"spectrum_data_*_multiline_{i}.csv")
            if matches:
                csv_files.append(matches[0])
        except subprocess.CalledProcessError as e:
            print(f"Error running spectrum_generator.py: {e}")
            break
    
    # Save the list of CSV files for validation
    with open("generated_csv_files.txt", "w") as f:
        for csvf in csv_files:
            f.write(csvf + "\n")
    
    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"CSV files generated: {len(csv_files)}/{n}")
    print(f"Files saved to: generated_csv_files.txt")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
