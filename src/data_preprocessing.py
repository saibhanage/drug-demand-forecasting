import pandas as pd
from pathlib import Path

def clean_sales_data(input_path, output_path):
    """
    Reads a sales CSV file, converts the 'datum' column to datetime,
    and saves the cleaned data to a new file.
    """
    try:
        # Read the dataset from the input path
        df = pd.read_csv(input_path)
        
        print(f"Cleaning {input_path.name}...")
        
        # Convert the 'datum' column to a proper datetime format
        if 'datum' in df.columns:
            df['datum'] = pd.to_datetime(df['datum'])
        
        # Save the cleaned DataFrame to the output path
        df.to_csv(output_path, index=False)
        print(f"Successfully cleaned and saved to {output_path}\n")
            
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.\n")
    except Exception as e:
        print(f"An error occurred while processing {input_path.name}: {e}\n")

# --- Main script execution ---

# Define the data directories using forward slashes
raw_data_dir = Path("drug-demand-forecasting/data/raw")
processed_data_dir = Path("drug-demand-forecasting/data/processed")

# Create the 'processed' directory if it doesn't exist
processed_data_dir.mkdir(parents=True, exist_ok=True)

# List of the datasets to be cleaned
files_to_clean = [
    'salesdaily.csv',
    'saleshourly.csv',
    'salesmonthly.csv',
    'salesweekly.csv'
]

# Loop through each file, define full paths, and apply the cleaning function
for filename in files_to_clean:
    # Construct the full path for the input file
    input_file = raw_data_dir / filename
    
    # Construct the full path for the output (cleaned) file
    output_file = processed_data_dir / ("cleaned_" + filename)
    
    # Run the cleaning function
    clean_sales_data(input_file, output_file)