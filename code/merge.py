import pandas as pd

# Load the weather and air quality data
weather_file = "data\weather_data.csv"
air_quality_file = "data\air_quality_data.csv"
output_file = "data\merged_output.csv"

# Read the CSV files
weather_df = pd.read_csv(weather_file)
air_quality_df = pd.read_csv(air_quality_file)

# Merge the data based on timestamp alignment
merged_df = pd.concat([weather_df, air_quality_df], axis=1)

# Save the merged data to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged data saved to {output_file}")