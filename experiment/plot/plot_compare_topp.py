#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare success rates across different parameter settings (p and T)

Usage:
    python plot_compare_parameters.py
    
Configuration:
    Edit PARAMETER_FOLDERS to define which parameter settings to compare
    
Output:
    - Comparison chart saved in ./figures_mlsys/ as parameter_comparison.pdf
"""

import re
import math
import glob
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Import plot utilities for unified styling
from plot_utils import setup_plot_font_style, get_color, convert_scientific_notation

# Set font for better rendering
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# Parameter setting to folder path mapping for comparison
# Label format: $p$=value, $T$=value matching parse_results_dashed.py
PARAMETER_FOLDERS = {
    "$p$=0.9, $T$=0.6": "/home/shuyilin/Jailbreak_oracle/result/20251018_033631_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "$p$=0.9, $T$=1.0": "/home/shuyilin/Jailbreak_oracle/result/20251018_200557_Llama-3.1-8B-Instruct_-1_1.0_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "$p$=0.8, $T$=1.0": "/home/shuyilin/Jailbreak_oracle/result/20251019_053455_Llama-3.1-8B-Instruct_-1_1.0_0.8_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    # Add more parameter combinations here
}

# Use unified color scheme from plot_utils
def get_unified_parameter_color(param_name):
    """Get color using unified color scheme from plot_utils."""
    # Extract p and T values from the parameter name
    if "$p$=0.9, $T$=0.6" in param_name:
        return get_color('topp', (0.9, 0.6))
    elif "$p$=0.9, $T$=1.0" in param_name:
        return get_color('topp', (0.9, 1.0))
    elif "$p$=0.8, $T$=1.0" in param_name:
        return get_color('topp', (0.8, 1.0))
    else:
        return '#000000'  # Default black color

def extract_time_from_filename(filepath):
    """Extract time (in seconds) from filename"""
    match = re.search(r'_t(\d+)s\.txt$', filepath)
    if match:
        return int(match.group(1))
    return None

def parse_main_file_safe_and_success(main_filepath):
    """Parse the main file to get Safe status, Running time, and success status for each Result"""
    result_info = {}
    
    with open(main_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find all Result separators
    result_indices = []
    for i, line in enumerate(lines):
        if line.strip().startswith('==================== Result'):
            result_num_match = re.search(r'Result (\d+)', line)
            if result_num_match:
                result_num = int(result_num_match.group(1))
                result_indices.append((i, result_num))
    
    # Parse Safe status, Running time, and success status for each Result
    total_success_count = 0
    for idx, (start_idx, result_num) in enumerate(result_indices):
        # Determine end position of current Result
        if idx < len(result_indices) - 1:
            end_idx = result_indices[idx + 1][0]
        else:
            end_idx = len(lines)
        
        is_safe = None
        running_time = None
        is_success = False
        
        # Find Safe status, Running time, and Success status for this Result
        for i in range(start_idx, end_idx):
            line = lines[i]
            if line.strip().startswith('Safe:'):
                safe_match = re.search(r'Safe:\s*(YES|NO)', line)
                if safe_match:
                    is_safe = (safe_match.group(1) == 'YES')
            if line.strip().startswith('Running time:'):
                time_match = re.search(r'Running time:\s*([\d.]+)\s*seconds', line)
                if time_match:
                    running_time = float(time_match.group(1))
            if line.strip().startswith('Success:'):
                success_match = re.search(r'Success:\s*(True|False)', line)
                if success_match:
                    is_success = (success_match.group(1) == 'True')
                    if is_success:
                        total_success_count += 1
        
        if is_safe is not None:
            result_info[result_num] = {
                'is_safe': is_safe,
                'running_time': running_time,
                'is_success': is_success
            }
    
    return result_info, total_success_count

def find_main_file(folder_path):
    """Find the main file (without time suffix) in the folder"""
    all_txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    # Filter out time snapshot files (those with _t\d+s pattern) and analysis files
    main_files = []
    for f in all_txt_files:
        if not re.search(r'_t\d+s\.txt$', f) and not re.search(r'_safe_tree_analysis\.txt$', f) and not re.search(r'completion_analysis\.txt$', f):
            main_files.append(f)
    
    if not main_files:
        return None
    
    # If multiple main files, choose the one with the longest name (most specific)
    return sorted(main_files, key=lambda x: len(x), reverse=True)[0]

def get_success_rates_over_time(folder_path):
    """Get success rates over time for a single parameter setting using main file running times"""
    main_file = find_main_file(folder_path)
    if not main_file:
        print(f"Error: Main file not found in {folder_path}")
        return None, None
    
    # Parse main file
    result_info, total_success_final = parse_main_file_safe_and_success(main_file)
    total_prompts = len(result_info)
    
    if total_prompts == 0:
        print(f"Error: No results found in {folder_path}")
        return None, None
    
    # Sort results by running time
    sorted_results = sorted(result_info.items(), key=lambda x: x[1]['running_time'] if x[1]['running_time'] is not None else float('inf'))
    
    # Calculate cumulative success rate at each completion time
    times = [0]  # Start from time 0
    success_rates = [0]  # Start with 0% success rate
    success_count = 0
    
    for result_num, info in sorted_results:
        if info['running_time'] is not None:
            if info['is_success']:
                success_count += 1
            
            times.append(info['running_time'])
            success_rates.append((success_count / total_prompts) * 100)
    
    return times, success_rates

def compare_parameters():
    """Create a comparison plot of success rates for different parameter settings"""
    print("="*80)
    print("Parameter Comparison - Success Rate Over Time")
    print("="*80)
    print(f"\nComparing {len(PARAMETER_FOLDERS)} parameter settings:")
    for i, param_name in enumerate(PARAMETER_FOLDERS.keys(), 1):
        print(f"  {i}. {param_name}")
    print()
    
    # Create plot
    plt.figure(figsize=(12, 6.5))
    
    # Set up unified font style
    setup_plot_font_style()
    
    # Plot each parameter setting
    for param_name, folder_path in PARAMETER_FOLDERS.items():
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
        
        color = get_unified_parameter_color(param_name)
        
        # Convert scientific notation in label if needed
        formatted_label = param_name  # Keep original format for p and T parameters
        
        plt.plot(times, success_rates, '-', linewidth=2.5, label=formatted_label, color=color)
        print(f"  Final success rate: {success_rates[-1]:.2f}% at {times[-1]}s")
    
    
    plt.xlabel('Search Time Per Query (seconds)')
    plt.ylabel('Jailbreak Discovery Rate (%)')
    plt.grid(True)
    plt.legend(loc='upper left')  # Put legend in upper left corner
    
    # Set y-axis to start from 0 and end at 100
    plt.ylim(bottom=0, top=100)
    
    # Set x-axis to linear scale with range 0-600
    plt.xlim(left=0, right=800)
    
    # Adjust tick label positioning to avoid overlap
    plt.tick_params(axis='x', pad=10)  # Move x-axis labels slightly further from axis
    plt.tick_params(axis='y', pad=8)  # Move y-axis labels slightly further from axis
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    output_dir = './figures_mlsys'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'topp_comparison.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"Comparison plot saved to: {os.path.abspath(output_path)}")
    print(f"{'='*80}\n")

def main():
    """Main function - compare all parameter settings defined in PARAMETER_FOLDERS"""
    compare_parameters()

if __name__ == "__main__":
    main()

