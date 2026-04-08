#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare success rates across different parameter settings (both top-p and top-k)

Usage:
    python plot_compare_combined.py
    
Configuration:
    Edit PARAMETER_FOLDERS and TOPK_FOLDERS to define which settings to compare
    
Output:
    - Combined comparison chart saved in ./figures_mlsys/ as combined_comparison.pdf
"""

import re
import math
import glob
import os
import re as re_module
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Import plot utilities for unified styling
from plot_utils import setup_plot_font_style, get_color

# Set font for better rendering
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# Parameter setting to folder path mapping for comparison (from topp script)
PARAMETER_FOLDERS = {
    "$p$=0.9, $T$=0.6": "/home/shuyi/BOA/results/merged_Llama-3.1-8B-Instruct",
    "$p$=0.8, $T$=1.0": "/home/shuyi/BOA/results/merged_llama3.1_p0.8",
    "$p$=0.9, $T$=1.0": "/home/shuyi/BOA/results/merged_llama3.1_p0.9",

}

# Top-k setting to folder path mapping for comparison (from topk script)
TOPK_FOLDERS = {
    "$k$=5": "/home/shuyi/BOA/results/merged_llama3.1_k5",
    "$k$=10": "/home/shuyi/BOA/results/merged_llama3.1_k10",
    "$k$=20": "/home/shuyi/BOA/results/merged_llama3.1_k20",
}

# Use unified color scheme from plot_utils
def get_unified_color(param_name):
    """Get color using unified color scheme from plot_utils."""
    if param_name == "$k$=5":
        return get_color('topk', 5)
    elif param_name == "$k$=10":
        return get_color('topk', 10)
    elif param_name == "$k$=20":
        return get_color('topk', 20)
    elif "$p$=0.9, $T$=0.6" in param_name:
        return get_color('topp', (0.9, 0.6))
    elif "$p$=0.9, $T$=1.0" in param_name:
        return get_color('topp', (0.9, 1.0))
    elif "$p$=0.8, $T$=1.0" in param_name:
        return get_color('topp', (0.8, 1.0))
    else:
        return '#000000'  # Default black color

# Line styles to distinguish between top-p and top-k
LINE_STYLES = {
    # Top-p settings use solid lines
    "$p$=0.9, $T$=0.6": '-',
    "$p$=0.9, $T$=1.0": '-', 
    "$p$=0.8, $T$=1.0": '-',
    
    # Top-k settings use dashed lines
    "$k$=5": '-',
    "$k$=10": '-',
    "$k$=20": '-',
}

def get_success_rates_over_time(folder_path):
    """Get success rates over time for a single parameter setting using TXT data"""
    txt_path = None
    for name in ['runs.txt', 'run.txt']:
        path = os.path.join(folder_path, name)
        if os.path.exists(path):
            txt_path = path
            break
    if not txt_path:
        print(f"Error: TXT file not found in {folder_path}")
        return None, None

    # Parse TXT file
    results = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re_module.split(r'={2,}\n', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        prompt_match = re_module.search(r'Prompt\s*\((\d+)/\d+\)', block)
        if not prompt_match:
            continue
        safe_match = re_module.search(r'^Safe:\s*(\S+)', block, re_module.MULTILINE)
        is_success = safe_match is not None and safe_match.group(1) == 'NO'
        time_match = re_module.search(r'^Elapsed time:\s*([\d.]+)s', block, re_module.MULTILINE)
        running_time = float(time_match.group(1)) if time_match else None
        results.append({
            'is_success': is_success,
            'running_time': running_time,
        })

    total_prompts = len(results)
    if total_prompts == 0:
        print(f"Error: No results found in {folder_path}")
        return None, None

    # Sort results by running time
    sorted_results = sorted(results, key=lambda x: x['running_time'] if x['running_time'] is not None else float('inf'))

    # Calculate cumulative success rate at each completion time
    times = [0]  # Start from time 0
    success_rates = [0]  # Start with 0% success rate
    success_count = 0

    for info in sorted_results:
        if info['running_time'] is not None:
            if info['is_success']:
                success_count += 1

            times.append(info['running_time'])
            success_rates.append((success_count / 128) * 100)

    return times, success_rates

def compare_combined():
    """Create a combined comparison plot of success rates for both top-p and top-k settings"""
    print("="*80)
    print("Combined Comparison - Success Rate Over Time (Top-p & Top-k)")
    print("="*80)
    
    # Combine all folders
    all_folders = {}
    all_folders.update(PARAMETER_FOLDERS)
    all_folders.update(TOPK_FOLDERS)
    
    print(f"\nComparing {len(all_folders)} parameter settings:")
    print("Top-p settings:")
    for i, param_name in enumerate(PARAMETER_FOLDERS.keys(), 1):
        print(f"  {i}. {param_name} (solid line)")
    print("Top-k settings:")
    for i, topk_name in enumerate(TOPK_FOLDERS.keys(), 1):
        print(f"  {len(PARAMETER_FOLDERS) + i}. {topk_name} (dashed line)")
    print()
    
    # Create plot
    plt.figure(figsize=(12, 6.5))
    
    # Set up unified font style
    setup_plot_font_style()
    
    # Plot each parameter setting
    for param_name, folder_path in all_folders.items():
        print(f"Processing {param_name}...")
        times, success_rates = get_success_rates_over_time(folder_path)
        
        if times is None or success_rates is None:
            print(f"  Skipping {param_name} due to errors")
            continue
        
        # Check for times > 1000 seconds
        times_over_1000 = [t for t in times if t > 1000]
        if times_over_1000:
            print(f"  WARNING: Found {len(times_over_1000)} time points > 1000s: {times_over_1000[:5]}...")
            print(f"  Max time: {max(times)}s")
        
        color = get_unified_color(param_name)
        linestyle = LINE_STYLES.get(param_name, '-')
        plt.plot(times, success_rates, linestyle, linewidth=2.5, label=param_name, color=color)
        print(f"  Final success rate: {success_rates[-1]:.2f}% at {times[-1]}s")
    
    
    plt.xlabel('Search Time Per Query (seconds)')
    plt.ylabel('Jailbreak Discovery Rate (%)')
    plt.grid(True)
    plt.legend(loc='upper left')  # Put legend in upper left corner
    
    # Set y-axis to start from 0 and end at 100
    plt.ylim(bottom=0, top=100)
    
    # Set x-axis to linear scale with range 0-800
    plt.xlim(left=0, right=600)
    
    # Adjust tick label positioning to avoid overlap
    plt.tick_params(axis='x', pad=10)  # Move x-axis labels slightly further from axis
    plt.tick_params(axis='y', pad=8)  # Move y-axis labels slightly further from axis
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    output_dir = './figures_mlsys'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_comparison.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"Combined comparison plot saved to: {os.path.abspath(output_path)}")
    print(f"{'='*80}\n")

def main():
    """Main function - compare all parameter settings (both top-p and top-k)"""
    compare_combined()

if __name__ == "__main__":
    main()
