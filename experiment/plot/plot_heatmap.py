import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

def parse_filename(filename):
    """Extract experiment parameters from filename.
    Expected format: MODELNAME_TOPK_TEMPERATURE_TOPP_EPSILON_output.txt
    """
    # Remove .txt extension and split by underscore
    parts = filename.replace('_output.txt', '').split('_')
    
    params = {}
    if len(parts) >= 5:
        params['model'] = parts[0]
        params['topk'] = float(parts[1])
        params['temperature'] = float(parts[2])
        params['topp'] = float(parts[3])
        params['epsilon'] = float(parts[4])
    
    return params

def load_results(base_dir):
    """Load all results from the specified directory."""
    results = []
    
    print(f"Looking for results in {base_dir}")
    
    for eps_dir in ['1e-0', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5']:
        eps_path = os.path.join(base_dir, eps_dir)
        if not os.path.exists(eps_path):
            print(f"Directory not found: {eps_path}")
            continue
            
        print(f"\nProcessing directory: {eps_path}")
        files_found = False
        
        for filename in os.listdir(eps_path):
            if not filename.endswith('_output.txt'):
                continue
                
            files_found = True
            filepath = os.path.join(eps_path, filename)
            print(f"\nProcessing file: {filename}")
            
            params = parse_filename(filename)
            print(f"Extracted parameters: {params}")
            
            # Read the output file
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Split content into individual search results
                search_results = content.split('###Prompt ')
                
                # Initialize prompt counter
                valid_prompt_idx = 1
                
                for result in search_results[1:]:  # Skip first empty split
                    # Skip if no nuanced_judger result
                    if 'nuanced_judger:' not in result.lower():
                        print(f"Skipping prompt: no nuanced_judger result")
                        continue
                        
                    # Extract search time
                    time_line = [line for line in result.split('\n') if 'search_time:' in line]
                    if time_line:
                        try:
                            time_str = time_line[0].split('search_time:')[1].strip()
                            time_to_success = float(time_str)
                            
                            # Check nuanced_judger result
                            is_unsafe = 'nuanced_judger: unsafe' in result.lower()
                            
                            results.append({
                                'prompt_idx': valid_prompt_idx,
                                'model': params.get('model', 'unknown'),
                                'topp': params.get('topp', 0),
                                'topk': params.get('topk', 0),
                                'temperature': params.get('temperature', 0),
                                'epsilon': params.get('epsilon', 0),
                                'time_to_success': time_to_success,
                                'is_unsafe': is_unsafe
                            })
                            print(f"Processed prompt {valid_prompt_idx}: time={time_to_success}, unsafe={is_unsafe}")
                            valid_prompt_idx += 1
                        except (IndexError, ValueError) as e:
                            print(f"Error extracting time: {e}")
                            continue
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        
        if not files_found:
            print(f"No output files found in {eps_path}")
    
    df = pd.DataFrame(results)
    print(f"\nTotal results loaded: {len(df)}")
    if len(df) > 0:
        print("\nSample of loaded data:")
        print(df.head())
        print("\nNumber of unsafe attempts:", df['is_unsafe'].sum())
    
    return df

def create_heatmap(df, output_file='jailbreak_heatmap.pdf'):
    """Create and save a heatmap visualization."""
    if len(df) == 0:
        print("No data to plot!")
        return
        
    # Create experiment labels
    df['experiment'] = df.apply(
        lambda x: f"{x['model']}_topk{x['topk']}_temp{x['temperature']}_topp{x['topp']}_eps{x['epsilon']}", 
        axis=1
    )
    
    # Create two separate dataframes for unsafe and safe results
    unsafe_df = df[df['is_unsafe']].copy()
    safe_df = df[~df['is_unsafe']].copy()
    
    # Get all unique experiments
    all_experiments = sorted(df['experiment'].unique())
    
    # Calculate unsafe count for each prompt and sort prompts
    prompt_unsafe_counts = df.groupby('prompt_idx')['is_unsafe'].sum()
    sorted_prompts = prompt_unsafe_counts.sort_values(ascending=False).index.tolist()
    
    # Create a single DataFrame for all results with float dtype
    combined_data = pd.DataFrame(index=all_experiments, columns=sorted_prompts, dtype=float)
    
    # Fill in unsafe results
    for _, row in unsafe_df.iterrows():
        combined_data.loc[row['experiment'], row['prompt_idx']] = float(row['time_to_success'])
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Create heatmap with log scale for unsafe results (green)
    sns.heatmap(
        combined_data,
        cmap='Greens_r',  # Inverted green colormap
        norm=LogNorm(),
        cbar_kws={'label': 'Time to Success (log scale)'},
        mask=combined_data.isna()
    )
    
    # Overlay safe results (single gray)
    if not safe_df.empty:
        # Create a mask for safe results
        safe_mask = pd.DataFrame(True, index=all_experiments, columns=sorted_prompts)
        for _, row in safe_df.iterrows():
            safe_mask.loc[row['experiment'], row['prompt_idx']] = False
        
        # Create a constant value array for safe results with float dtype
        safe_values = pd.DataFrame(0.5, index=all_experiments, columns=sorted_prompts, dtype=float)
        
        # Plot safe results with a single gray color
        sns.heatmap(
            safe_values,
            cmap=plt.cm.Greys,
            vmin=0,
            vmax=1,
            mask=safe_mask,
            alpha=0.5,
            cbar=False
        )
    
    plt.title('Jailbreak Results Heatmap\n(Green = Unsafe, Gray = Safe)')
    plt.xlabel('Prompt Index (sorted by unsafe count)')
    plt.ylabel('Experiment Configuration')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"\nHeatmap saved to {output_file}")

def main():
    base_dir = 'result'
    df = load_results(base_dir)
    
    # Save results to CSV
    csv_output = 'jailbreak_results.csv'
    df.to_csv(csv_output, index=False)
    print(f"\nResults saved to {csv_output}")
    
    create_heatmap(df)

if __name__ == '__main__':
    main() 