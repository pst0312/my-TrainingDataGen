import pandas as pd
import plotly.express as px
import os

# Read the list of generated CSV files
csv_list_path = "generated_csv_files.txt"
if not os.path.exists(csv_list_path):
	print(f"CSV list file not found: {csv_list_path}")
	exit(1)
with open(csv_list_path, "r") as f:
	csv_files = [line.strip() for line in f if line.strip()]

for csv_file in csv_files:
	print(f"\nValidating: {csv_file}")
	df = pd.read_csv(csv_file)
	print(df.head(10))
	chart = px.line(df, x="energy", y="intensity", color="line_id", title=f"Validation: {csv_file}")
	chart.show()