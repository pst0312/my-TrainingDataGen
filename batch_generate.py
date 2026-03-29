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

    for i in range(1, n + 1):
        print(f"\n--- Generating spectrum {i} of {n} ---")
        try:
            subprocess.run([sys.executable, "spectrum_generator.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running spectrum_generator.py: {e}")
            break

if __name__ == "__main__":
    main()
