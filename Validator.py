import pandas as pd
import plotly.express as px

test_case = "spectrum_data_xps_multiline.csv"
# We store the data in 'df' (Data Frame)
df = pd.read_csv(test_case)

# We print the first 10 rows so we can see it worked
print(df.head(10))

chart = px.line(df, x="energy", y="intensity", color="line_id", title="XPS Spectrum (Multi-Line)")
chart.show()    