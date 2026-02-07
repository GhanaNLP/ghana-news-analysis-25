import pandas as pd
import os
from pathlib import Path

def merge_csvs_by_phrase(folder_path, output_file="merged_phrases.csv"):
    """
    Merge all CSV files in a folder, combining rows with duplicate phrases
    (case-insensitive) by summing their counts.
    """
    folder = Path(folder_path)
    
    # Find all CSV files
    csv_files = list(folder.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")
    
    print(f"Found {len(csv_files)} CSV files to merge...")
    
    # Read and combine all CSVs
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add source column if not present
            if 'csv_source' not in df.columns:
                df['csv_source'] = csv_file.stem
            
            # Normalize phrase to lowercase for comparison (keep original in display)
            df['phrase_lower'] = df['phrase'].str.lower()
            
            all_dfs.append(df)
            print(f"  ✓ Loaded {csv_file.name}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file.name}: {e}")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows before merging: {len(combined_df)}")
    
    # Group by lowercase phrase and POS, sum the counts, aggregate sources
    # Keep the first occurrence's case as the canonical form
    merged_df = combined_df.groupby(['phrase_lower', 'pos'], as_index=False).agg({
        'phrase': 'first',  # Keep first occurrence's casing
        'count': 'sum',
        'csv_source': lambda x: ', '.join(sorted(set(x)))
    })
    
    # Reorder columns to match original format
    merged_df = merged_df[['phrase', 'pos', 'count', 'csv_source']]
    
    # Sort by count descending (most frequent first)
    merged_df = merged_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    print(f"Total rows after merging duplicates: {len(merged_df)}")
    print(f"Duplicates merged: {len(combined_df) - len(merged_df)} rows")
    
    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved merged result to: {output_file}")
    
    return merged_df

# Example usage
if __name__ == "__main__":
    # Replace with your folder path
    FOLDER_PATH = "/home/owusus/Documents/GitHub/ghana-news-analysis/noun-phrases/"  # Change this to your folder path
    OUTPUT_FILE = "/home/owusus/Documents/GitHub/ghana-news-analysis/merged_phrases.csv"
    
    # Run the merge
    result = merge_csvs_by_phrase(FOLDER_PATH, OUTPUT_FILE)
    
    # Display top 10 results
    print("\nTop 10 phrases by total count:")
    print(result.head(10).to_string())
