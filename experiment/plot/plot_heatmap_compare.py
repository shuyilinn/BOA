#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create heatmap comparing jailbreak results across different settings

Usage:
    python plot_heatmap_compare.py
    
Configuration:
    Edit SETTINGS_FOLDERS to define which settings to compare
    
Output:
    - Heatmap saved in ./figures_mlsys/ as results_heatmap.pdf
"""

import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import re as re_module

# Import plot utilities for unified styling
from plot_utils import setup_plot_font_style

# Setting to folder path mapping for comparison
SETTINGS_FOLDERS = {



    "top-$p$ ($p$=0.9), T=0.6": "/home/shuyi/BOA/results/merged_Llama-3.1-8B-Instruct",
        "top-$p$ ($p$=0.8), T=1.0": "/home/shuyi/BOA/results/merged_llama3.1_p0.8",
    "top-$p$ ($p$=0.9), T=1.0": "/home/shuyi/BOA/results/merged_llama3.1_p0.9",


        "top-$k$ ($k$=5), T=1.0": "/home/shuyi/BOA/results/merged_llama3.1_k5",
    "top-$k$ ($k$=10), T=1.0": "/home/shuyi/BOA/results/merged_llama3.1_k10",
        "top-$k$ ($k$=20), T=1.0": "/home/shuyi/BOA/results/merged_llama3.1_k20",
}

def find_txt_file(folder_path):
    """Find the TXT data file in the folder"""
    for name in ['runs.txt', 'run.txt']:
        path = os.path.join(folder_path, name)
        if os.path.exists(path):
            return path
    return None

def parse_txt_file(txt_filepath):
    """Parse the TXT file to get results for each prompt"""
    results = []

    with open(txt_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into blocks by separator lines
    blocks = re_module.split(r'={2,}\n', content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract prompt index: "Prompt (X/Y):"
        prompt_match = re_module.search(r'Prompt\s*\((\d+)/\d+\)', block)
        if not prompt_match:
            continue
        prompt_idx = int(prompt_match.group(1)) - 1  # convert to 0-based

        # Extract Safe field (None or placeholder values are treated as safe)
        safe_match = re_module.search(r'^Safe:\s*(\S+)', block, re_module.MULTILINE)
        is_unsafe = safe_match is not None and safe_match.group(1) == 'NO'

        # Extract elapsed time
        time_match = re_module.search(r'^Elapsed time:\s*([\d.]+)s', block, re_module.MULTILINE)
        elapsed_time = float(time_match.group(1)) if time_match else 0

        results.append({
            'prompt_idx': prompt_idx,
            'is_unsafe': is_unsafe,
            'is_success': is_unsafe,
            'time_to_success': elapsed_time,
        })

    return results

def load_all_results():
    """Load results from all configured folders"""
    all_results = []
    
    print("="*80)
    print("Loading results from all settings")
    print("="*80)
    
    for setting_name, folder_path in SETTINGS_FOLDERS.items():
        print(f"\nProcessing {setting_name}...")
        print(f"  Folder: {folder_path}")
        
        txt_file = find_txt_file(folder_path)
        if not txt_file:
            print(f"  ERROR: TXT file not found")
            continue

        print(f"  TXT file: {os.path.basename(txt_file)}")

        results = parse_txt_file(txt_file)
        print(f"  Loaded {len(results)} prompts")
        
        unsafe_count = sum(1 for r in results if r['is_unsafe'])
        success_count = sum(1 for r in results if r['is_success'])
        print(f"  Unsafe: {unsafe_count}, Success: {success_count}")
        
        # Add setting name to each result
        for result in results:
            result['setting'] = setting_name
        
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    print(f"\n{'='*80}")
    print(f"Total results loaded: {len(df)}")
    print(f"{'='*80}\n")
    
    return df

def create_heatmap(df, output_file='results_heatmap.pdf'):
    """Create and save a heatmap visualization."""
    if len(df) == 0:
        print("No data to plot!")
        return
    
    # Get all unique settings and prompts - fixed order: topp=0.6 at top, then topp=0.8, 0.9, finally k values ascending
    all_settings = list(SETTINGS_FOLDERS.keys())
    
    # Custom sort: topp=0.6 at top, then topp=0.8, 0.9, finally k values ascending
    def custom_sort_key(setting):
        if "top-$p$" in setting:
            # top-p strategy, sort by p value
            p_match = re.search(r'p=([\d.]+)', setting)
            p_value = float(p_match.group(1)) if p_match else 0
            if p_value == 0.6:
                return (0, 0)  # topp=0.6 at top
            elif p_value == 0.8:
                return (1, 0)  # topp=0.8 second
            elif p_value == 0.9:
                return (2, 0)  # topp=0.9 third
            else:
                return (3, -p_value)  # Other p values sorted by value
        elif "top-$k$" in setting:
            # top-k strategy, sort by k value ascending
            k_match = re.search(r'\$k\$=(\d+)', setting)
            k_value = int(k_match.group(1)) if k_match else 0
            return (4, k_value)  # 4 means top-k, k values ascending
        else:
            return (5, 0)  # Other cases at bottom
    
    all_settings.sort(key=custom_sort_key)
    all_prompts = sorted(df['prompt_idx'].unique())
    
    # Calculate unsafe count for each prompt across all settings
    prompt_unsafe_counts = df.groupby('prompt_idx')['is_unsafe'].sum()
    sorted_prompts = prompt_unsafe_counts.sort_values(ascending=False).index.tolist()
    
    # Create DataFrame for unsafe results
    unsafe_df = df[df['is_unsafe']].copy()

    # Create a DataFrame for unsafe results (time values) with float dtype
    combined_data = pd.DataFrame(index=all_settings, columns=sorted_prompts, dtype=float)
    
    # Fill in unsafe results
    for _, row in unsafe_df.iterrows():
        combined_data.loc[row['setting'], row['prompt_idx']] = float(row['time_to_success'])
    
    # Calculate figure size to make cells rectangular (elongated)
    n_rows = len(all_settings)
    n_cols = len(sorted_prompts)
    cell_width = 0.25   # Narrower width for elongated heatmap
    cell_height = 0.4   # Keep appropriate height
    
    fig_width = n_cols * cell_width    # Slightly reduce overall width
    fig_height = n_rows * cell_height + 1.5  # Based on height
    
    # Set up unified font style first
    setup_plot_font_style()
    
    # Simplified to single subplot, no gridspec needed
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap with log scale for unsafe results (green)
    # Use log distribution of natural data range, reference plot_heatmap.py
    if combined_data.isna().all().all():
        print("Warning: No valid data for heatmap")
        return
    
    print(f"Using natural data range for log scale")
    
    # Simplified heatmap with colorbar
    from matplotlib.colors import LogNorm
    
    # Create heatmap with colorbar, adjust colorbar position
    main_hm = sns.heatmap(
        combined_data,
        ax=ax,
        cmap='Blues_r',  # 0 is dark blue, 1000 is white (reversed)
        norm=LogNorm(),  # Use natural data range, reference plot_heatmap.py
        cbar=True,
        cbar_kws={
            'label': 'Search Time(s)', 
            'shrink': 1.0,  # Make colorbar same height as heatmap
            'aspect': 12,   # Reduce aspect to make colorbar wider, fill right space
            'pad': 0.01    # Smaller gap to bring colorbar closer to heatmap
        },
        mask=combined_data.isna(),
        linewidths=0,  # Ultra-thin grid lines, nearly invisible
        linecolor='white'
    )
    
    # On natural log distribution, annotate 0 and 600
    import numpy as np
    cbar = main_hm.collections[0].colorbar
    
    # Get current colorbar range
    vmin, vmax = cbar.mappable.get_clim()
    print(f"Natural colorbar range: {vmin:.2f} to {vmax:.2f}")
    
    # Annotate 0, 100, and 600 (use minimum value to represent 0 in log scale)
    tick_positions = [vmin, 100, 600]  # Use data minimum for 0, add intermediate value 100
    tick_labels = ['0', '100', '600']
    
    # Only keep ticks within actual data range
    valid_ticks = []
    valid_labels = []
    for pos, label in zip(tick_positions, tick_labels):
        if vmin <= pos <= vmax:
            valid_ticks.append(pos)
            valid_labels.append(label)
    
    if valid_ticks:
        cbar.set_ticks(valid_ticks)
        cbar.set_ticklabels(valid_labels)
        print(f"Setting colorbar ticks: {valid_ticks} -> {valid_labels}")

    # Overlay safe/missing results (gray)
    # Any cell that is NOT unsafe (including missing data) is treated as safe and shown in gray
    safe_mask = ~combined_data.isna()  # True (masked) where unsafe data exists, False (shown) where NaN

    if not safe_mask.all().all():  # If there are any non-unsafe cells
        # Create a constant value array for safe/missing results
        safe_values = pd.DataFrame(0.5, index=all_settings, columns=sorted_prompts, dtype=float)

        # Second layer: gray safe/missing sample overlay, does not affect independent colorbar
        overlay = sns.heatmap(
            safe_values,
            ax=ax,
            cmap=plt.cm.Greys,
            vmin=0,
            vmax=1,
            mask=safe_mask,
            alpha=0.3,
            cbar=False,  # Do not create new colorbar
            linewidths=0.00001,  # Same ultra-thin lines as main heatmap

            linecolor='gray'  # Same gray as main heatmap
        )
    
    # Set axis labels and ticks - keep only axis names, remove all ticks and numbers
    ax.set_xlabel('Prompts')
    ax.set_ylabel('')
    
    # Remove all x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Remove tick marks (the small vertical lines) but keep labels
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    
    # Add red border feature: for p=0.9,T=0.6, add red border if it is the only safe result
    target_setting = "top-$p$ ($p$=0.9), T=0.6"
    if target_setting in all_settings:
        # Find target setting position on Y axis
        target_row_idx = all_settings.index(target_setting)

        # Check each prompt (column)
        for col_idx, prompt_idx in enumerate(sorted_prompts):
            # Check how many safe/missing results in this column
            safe_count = 0
            target_is_safe = False

            for setting in all_settings:
                # A cell is "safe" if it's not unsafe (either explicitly safe or missing data)
                is_unsafe = not pd.isna(combined_data.loc[setting, prompt_idx])
                if not is_unsafe:
                    safe_count += 1
                    if setting == target_setting:
                        target_is_safe = True

            # If only target setting is safe (safe_count=1 and target_is_safe=True)
            if safe_count == 1 and target_is_safe:
                from matplotlib.patches import Rectangle
                # Add red border
                rect = Rectangle((col_idx, target_row_idx), 1, 1, 
                               linewidth=1.5, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                print(f"Added red border for prompt {prompt_idx} at {target_setting}")
    
    # Use fig.tight_layout() and save without bbox_inches='tight'
    fig.tight_layout()
    
    output_dir = './figures_mlsys'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.1)  # Auto-trim whitespace
    plt.close(fig)
    
    print(f"\n{'='*80}")
    print(f"Heatmap saved to {os.path.abspath(output_path)}")
    print(f"{'='*80}\n")

def main():
    """Main function - create heatmap for all settings"""
    df = load_all_results()
    
    if len(df) > 0:
        # Save results to CSV
        csv_output = './figures_mlsys/jailbreak_results.csv'
        df.to_csv(csv_output, index=False)
        print(f"Results saved to {csv_output}\n")
        
        create_heatmap(df)
    else:
        print("No data loaded, skipping heatmap creation")

if __name__ == '__main__':
    main()

