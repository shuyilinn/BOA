#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare success rates across different epsilon settings

Usage:
    python plot_compare_epsilon.py
    
Configuration:
    Edit EPSILON_FOLDERS to define which epsilon settings to compare
    
Output:
    - Comparison chart saved in ./figures_mlsys/ as epsilon_comparison.pdf
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
from plot_utils import setup_plot_font_style, get_color, convert_scientific_notation

# Set font for better rendering
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# Epsilon setting to folder path mapping for comparison
# Label format: $\varepsilon$=value matching parse_result_dashed_different_epi.py
EPSILON_FOLDERS = {
     "$\\varepsilon$=1e-0": "/home/shuyi/BOA/results/merged_llama-3.1-lk1",
    #"$\\varepsilon$=1e-2": "/home/shuyi/BOA/results/merged_llama-3.1-lk0.01",
    "$\\varepsilon$=1e-4": "/home/shuyi/BOA/results/merged_Llama-3.1-8B-Instruct",
    "$\\varepsilon$=1e-8":"/home/shuyi/BOA/results/merged_llama-3.1-lk1e-8"
 
    # Add more epsilon values here
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

def get_success_rates_over_time(folder_path):
    """Get success rates over time for a single epsilon setting using runs.txt"""
    txt_path = None
    for name in ['runs.txt', 'run.txt']:
        path = os.path.join(folder_path, name)
        if os.path.exists(path):
            txt_path = path
            break
    if not txt_path:
        print(f"Error: txt file not found in {folder_path}")
        return None, None

    # Parse runs.txt
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
        # Extract Safe field (None or placeholder values are treated as safe)
        safe_match = re_module.search(r'^Safe:\s*(\S+)', block, re_module.MULTILINE)
        jailbreak_found = safe_match is not None and safe_match.group(1) == 'NO'
        # Extract elapsed time
        time_match = re_module.search(r'^Elapsed time:\s*([\d.]+)s', block, re_module.MULTILINE)
        duration = float(time_match.group(1)) if time_match else None
        results.append({
            'duration': duration,
            'jailbreak_found': jailbreak_found,
        })

    total_prompts = 128
    if len(results) == 0:
        print(f"Error: No results found in {folder_path}")
        return None, None

    # Sort results by duration
    results.sort(key=lambda x: x['duration'] if x['duration'] is not None else float('inf'))

    # Calculate cumulative success rate at each completion time
    times = [0]  # Start from time 0
    success_rates = [0]  # Start with 0% success rate
    success_count = 0

    for info in results:
        if info['duration'] is not None:
            if info['jailbreak_found']:
                success_count += 1

            times.append(info['duration'])
            success_rates.append((success_count / total_prompts) * 100)

    return times, success_rates

def compare_epsilon():
    """Create a comparison plot of success rates for different epsilon settings"""
    print("="*80)
    print("Epsilon Comparison - Success Rate Over Time")
    print("="*80)
    print(f"\nComparing {len(EPSILON_FOLDERS)} epsilon settings:")
    for i, epsilon_name in enumerate(EPSILON_FOLDERS.keys(), 1):
        print(f"  {i}. {epsilon_name}")
    print()
    
    # Create plot
    plt.figure(figsize=(12, 6.5))
    
    # Set up unified font style
    setup_plot_font_style()
    
    # Plot each epsilon setting
    for epsilon_name, folder_path in EPSILON_FOLDERS.items():
        print(f"Processing {epsilon_name}...")
        times, success_rates = get_success_rates_over_time(folder_path)
        
        if times is None or success_rates is None:
            print(f"  Skipping {epsilon_name} due to errors")
            continue
        
        # Check for times > 1000 seconds
        times_over_1000 = [t for t in times if t > 1000]
        if times_over_1000:
            print(f"  WARNING: Found {len(times_over_1000)} time points > 1000s: {times_over_1000[:5]}...")
            print(f"  Max time: {max(times)}s")
        
        color = get_unified_epsilon_color(epsilon_name)
        
        # Convert scientific notation in label
        formatted_label = convert_scientific_notation(epsilon_name)
        
        plt.plot(times, success_rates, '-', linewidth=2.5, label=formatted_label, color=color)
        print(f"  Final success rate: {success_rates[-1]:.2f}% at {times[-1]}s")
    
    
    plt.xlabel('Search Time Per Query (seconds)')
    plt.ylabel('Jailbreak Discovery Rate (%)')
    plt.grid(True)
    plt.legend(loc='upper left')  # Put legend in upper left corner
    
    # Set y-axis to start from 0 and end at 100
    plt.ylim(bottom=0, top=40)
    plt.yticks([0, 10, 20,30,40])

    # Set x-axis to linear scale with range 0-600
    plt.xlim(left=0, right=600)
    
    # Adjust tick label positioning to avoid overlap
    plt.tick_params(axis='x', pad=10)  # Move x-axis labels slightly further from axis
    plt.tick_params(axis='y', pad=8)  # Move y-axis labels slightly further from axis
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    output_dir = './figures_mlsys'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'epsilon_comparison.pdf')
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

