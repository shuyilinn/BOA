#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare success rates across different epsilon settings with Telescopic Logic

Usage:
    python plot_compare_epsilon_telescopic.py
    
Configuration:
    Edit EPSILON_FOLDERS to define which epsilon settings to compare
    
Output:
    - Comparison chart saved in ./figures_mlsys/ as epsilon_comparison_telescopic.pdf
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


# Epsilon setting to folder path mapping for comparison
# Label format: $\varepsilon$=value matching parse_result_dashed_different_epi.py
EPSILON_FOLDERS = {
    "$\\varepsilon$=1e-0": "/home/shuyilin/Jailbreak_oracle/result/20251020_023559_Llama-3.1-8B-Instruct_-1_0.6_0.9_1.0_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "$\\varepsilon$=1e-2": "/home/shuyilin/Jailbreak_oracle/result/20251020_164738_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.01_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "$\\varepsilon$=1e-4": "/home/shuyilin/Jailbreak_oracle/result/20251018_033631_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "$\\varepsilon$=1e-2 (Telescopic)": "/home/shuyilin/Jailbreak_oracle/result/20251020_164738_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.01_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "$\\varepsilon$=1e-4 (Telescopic)": "/home/shuyilin/Jailbreak_oracle/result/20251018_033631_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
}

# Use unified color scheme from plot_utils
def get_unified_epsilon_color(epsilon_name):
    """Get color using unified color scheme from plot_utils."""
    # Extract epsilon value from the name
    if "1e-0" in epsilon_name:
        return get_color('epsilon', 1)  # 1e0
    elif "1e-2" in epsilon_name:
        return get_color('epsilon', 1e-2)
    elif "1e-4" in epsilon_name:
        return get_color('epsilon', 1e-4)
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

def analyze_epsilon_cascading_strategy(result_info, current_folder_path, label_name=""):
    """Analyze results using a cascading strategy similar to parse_result_dashed_different_epi.py"""
    
    # Extract epsilon value from folder path - look for the epsilon value at the end
    folder_name = os.path.basename(current_folder_path)
    
    # Try different patterns to extract epsilon
    if '_1.0_' in folder_name:
        current_epsilon = 1.0
    elif '_0.01_' in folder_name:
        current_epsilon = 0.01
    elif '_0.0001_' in folder_name:
        current_epsilon = 0.0001
    else:
        return result_info
    
    # Define available epsilon values based on actual data
    available_epsilons = [1.0, 0.01, 0.0001]  # Only the ones we have data for
    
    # Get epsilon values to search (all values >= current_epsilon)
    epsilons_to_search = [eps for eps in available_epsilons if eps >= current_epsilon]
    
    # Check if this is a telescopic line
    is_telescopic_line = "(Telescopic)" in label_name
    
    if is_telescopic_line and len(epsilons_to_search) > 1:
        print(f"  Applying cascading strategy for epsilon {current_epsilon} (Telescopic line)")
        print(f"  Searching epsilons: {epsilons_to_search}")
        
        telescopic_count = 0
        # For each prompt, check if it succeeded in any of the higher epsilon values
        for result_num, current_info in result_info.items():
            if not current_info['is_success']:
                # Check higher epsilon values for this prompt
                for higher_epsilon in epsilons_to_search[:-1]:  # Exclude current epsilon
                    higher_folder = find_folder_by_epsilon(higher_epsilon)
                    if higher_folder:
                        higher_result_info, _ = parse_main_file_safe_and_success(find_main_file(higher_folder))
                        if higher_result_info and result_num in higher_result_info:
                            if higher_result_info[result_num]['is_success']:
                                result_info[result_num]['is_success'] = True
                                telescopic_count += 1
                                print(f"    Prompt {result_num}: cascading success from epsilon {higher_epsilon}")
                                break  # Found success, no need to check further
        
        print(f"  Total telescopic successes: {telescopic_count}")
    
    return result_info

def find_folder_by_epsilon(target_epsilon):
    """Find folder path for a specific epsilon value"""
    for label, folder_path in EPSILON_FOLDERS.items():
        # Extract epsilon from label
        epsilon_match = re.search(r'\\varepsilon\$=1e-(\d+)', label)
        if epsilon_match:
            epsilon_power = int(epsilon_match.group(1))
            epsilon_value = 10 ** (-epsilon_power)
            if abs(epsilon_value - target_epsilon) < 1e-10:
                return folder_path
    return None

def get_success_rates_over_time(folder_path, label_name=""):
    """Get success rates over time for a single epsilon setting using main file running times"""
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
    
    # Apply cascading strategy for telescopic lines
    result_info = analyze_epsilon_cascading_strategy(result_info, folder_path, label_name)
    
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

def compare_epsilon():
    """Create a comparison plot of success rates for different epsilon settings"""
    print("="*80)
    print("Epsilon Comparison - Success Rate Over Time (with Telescopic Logic)")
    print("="*80)
    print(f"\nComparing {len(EPSILON_FOLDERS)} epsilon settings:")
    for i, epsilon_name in enumerate(EPSILON_FOLDERS.keys(), 1):
        print(f"  {i}. {epsilon_name}")
    print("\nTelescopic Logic (Cascading Strategy):")
    print("  - For epsilon=1e-4 (Telescopic): includes successes from 1e-0, 1e-1, 1e-2, 1e-3, 1e-4")
    print("  - For epsilon=1e-2 (Telescopic): includes successes from 1e-0, 1e-1, 1e-2")
    print("  - Regular lines: show original results without cascading")
    print()
    
    # Create plot
    plt.figure(figsize=(12, 6.5))
    
    # Set up unified font style
    setup_plot_font_style()
    
    # Plot each epsilon setting
    for epsilon_name, folder_path in EPSILON_FOLDERS.items():
        print(f"Processing {epsilon_name}...")
        times, success_rates = get_success_rates_over_time(folder_path, epsilon_name)
        
        if times is None or success_rates is None:
            print(f"  Skipping {epsilon_name} due to errors")
            continue
        
        # Check for times > 1000 seconds
        times_over_1000 = [t for t in times if t > 1000]
        if times_over_1000:
            print(f"  WARNING: Found {len(times_over_1000)} time points > 1000s: {times_over_1000[:5]}...")
            print(f"  Max time: {max(times)}s")
        
        color = get_unified_epsilon_color(epsilon_name)
        
        # Use dashed line for telescopic lines
        linestyle = '--' if "(Telescopic)" in epsilon_name else '-'
        
        # Convert scientific notation in label and format telescopic labels
        if "(Telescopic)" in epsilon_name:
            # Extract epsilon part and reformat for telescopic
            epsilon_part = epsilon_name.replace(" (Telescopic)", "")
            epsilon_part = convert_scientific_notation(epsilon_part)
            formatted_label = f"Telescopic Search({epsilon_part})"
        else:
            formatted_label = convert_scientific_notation(epsilon_name)
        
        plt.plot(times, success_rates, linestyle, linewidth=2.5, label=formatted_label, color=color)
        print(f"  Final success rate: {success_rates[-1]:.2f}% at {times[-1]}s")
    
    
    plt.xlabel('Search Time Per Query (seconds)')
    plt.ylabel('Jailbreak Discovery Rate (%)')
    plt.grid(True)
    plt.legend(loc='upper left')  # Put legend in upper left corner
    
    # Set y-axis to start from 0 and end at 100
    plt.ylim(bottom=0, top=60)
    
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
    output_path = os.path.join(output_dir, 'epsilon_comparison_telescopic.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"Comparison plot saved to: {os.path.abspath(output_path)}")
    print(f"{'='*80}\n")

def main():
    """Main function - compare all epsilon settings defined in EPSILON_FOLDERS"""
    compare_epsilon()

if __name__ == "__main__":
    main()
