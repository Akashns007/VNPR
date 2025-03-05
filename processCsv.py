import pandas as pd

def final_csv(input_file: str, output_file = "output/final_csv.csv"):
    # Load CSV file
    df = pd.read_csv(input_file)
    
    # Convert confidence values to numeric (in case of string format)
    df['license_number_score'] = pd.to_numeric(df['license_number_score'], errors='coerce')
    
    # Drop duplicates by keeping only the row with the highest confidence per car_id
    df_filtered = df.loc[df.groupby('car_id')['license_number_score'].idxmax()]
    
    # Save the cleaned data
    df_filtered.to_csv(output_file, index=False)
    
    print(f"Filtered data saved to {output_file}")
    
    return output_file


if __name__ == '__main__':
    final_csv("output/test_interpolated.csv")