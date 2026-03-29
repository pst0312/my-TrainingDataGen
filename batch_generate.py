import subprocess
import sys


def main():
    try:
        n = int(input("How many spectra would you like to generate? "))
    except Exception:
        print("Invalid input. Please enter an integer.")
        sys.exit(1)
    if n < 1:
        print("Please enter a positive integer.")
        sys.exit(1)

    csv_files = []
    for i in range(1, n + 1):
        print(f"\n--- Generating spectrum {i} of {n} ---")
        try:
            subprocess.run([sys.executable, "spectrum_generator.py", str(i)], check=True)
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
    print(f"\nCSV files generated: {csv_files}")

if __name__ == "__main__":
    main()
