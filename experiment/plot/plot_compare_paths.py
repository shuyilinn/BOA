#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare total paths (leaf nodes) across different settings

Usage:
    python plot_compare_paths.py
    
Configuration:
    Edit SETTINGS_FOLDERS to define which settings to compare
    
Output:
    - Comparison chart saved in ./figures_mlsys/ as paths_comparison.pdf
"""

import re
import math
import glob
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set font for better rendering
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# Setting to folder path mapping for comparison
SETTINGS_FOLDERS = {
    "Llama 3.1 (8B)": "/home/shuyilin/Jailbreak_oracle/result/20251018_033631_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "Llama 2 (7B)": "/home/shuyilin/Jailbreak_oracle/result/20251017_031711_Llama-2-7b-chat-hf_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "Vicuna 1.5 (7B)": "/home/shuyilin/Jailbreak_oracle/result/20251016_052222_vicuna-7b-v1.5_-1_0.9_0.6_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "Qwen3 (8B)": "/home/shuyilin/Jailbreak_oracle/result/20251021_063846_Qwen3-8B_20_0.6_0.95_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
}

# Color mapping
SETTINGS_COLORS = {
    "Llama 3.1 (8B)": "#2E86AB",
    "Llama 2 (7B)": "#F18F01",
    "Vicuna 1.5 (7B)": "#06A77D",
    "Qwen3 (8B)": "#B56576",
}

def extract_time_from_filename(filepath):
    """Extract time (in seconds) from filename"""
    match = re.search(r'_t(\d+)s\.txt$', filepath)
    if match:
        return int(match.group(1))
    return None

def parse_tree_file(filepath):
    """Parse tree structure file, group by prompt"""
    all_results = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find all Result separators
    result_indices = []
    for i, line in enumerate(lines):
        if line.strip().startswith('==================== Result') or \
           line.strip().startswith('==================== Prompt'):
            result_num_match = re.search(r'(?:Result|Prompt) (\d+)', line)
            if result_num_match:
                result_num = int(result_num_match.group(1))
                result_indices.append((i, result_num))
    
    # Parse tree for each Result
    for idx, (start_idx, result_num) in enumerate(result_indices):
        if idx < len(result_indices) - 1:
            end_idx = result_indices[idx + 1][0]
        else:
            end_idx = len(lines)
        
        # Find Tree in this Result
        tree_start = None
        for i in range(start_idx, end_idx):
            line = lines[i]
            if line.strip().startswith('Tree:'):
                tree_start = i + 1
                break
        
        if tree_start is None:
            continue
        
        # Parse tree nodes for this Result
        nodes = []
        pattern = r"- \[(\w+)\] '([^']*)' \(id: (\d+)\), logP: ([-\d.]+), score: ([-\d.]+)"
        
        for i in range(tree_start, end_idx):
            line = lines[i]
            if line.strip().startswith('===') or not line.strip():
                continue
            
            indent = (len(line) - len(line.lstrip())) // 2
            match = re.search(pattern, line)
            if match:
                status, token, node_id, log_p, score = match.groups()
                nodes.append({
                    'status': status,
                    'token': token,
                    'id': int(node_id),
                    'logP': float(log_p),
                    'score': float(score),
                    'depth': indent,
                    'line_num': i
                })
        
        if nodes:
            all_results.append({
                'result_num': result_num,
                'nodes': nodes
            })
    
    return all_results

def count_leaf_nodes(nodes):
    """Count total leaf nodes (paths)"""
    leaf_count = 0
    
    for i, node in enumerate(nodes):
        # Check if it's a leaf node: all subsequent nodes have depth <= current node
        is_leaf = True
        for j in range(i + 1, len(nodes)):
            if nodes[j]['depth'] > node['depth']:
                is_leaf = False
                break
            elif nodes[j]['depth'] <= node['depth']:
                break
        
        if is_leaf:
            leaf_count += 1
    
    return leaf_count

def parse_main_file_safe_and_success(main_filepath):
    """Parse the main file to get Safe status, Running time, and success status"""
    result_info = {}
    
    with open(main_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    result_indices = []
    for i, line in enumerate(lines):
        if line.strip().startswith('==================== Result'):
            result_num_match = re.search(r'Result (\d+)', line)
            if result_num_match:
                result_num = int(result_num_match.group(1))
                result_indices.append((i, result_num))
    
    total_success_count = 0
    for idx, (start_idx, result_num) in enumerate(result_indices):
        if idx < len(result_indices) - 1:
            end_idx = result_indices[idx + 1][0]
        else:
            end_idx = len(lines)
        
        is_safe = None
        running_time = None
        is_success = False
        
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
    
    main_files = []
    for f in all_txt_files:
        if not re.search(r'_t\d+s\.txt$', f):
            main_files.append(f)
    
    if not main_files:
        return None
    
    return sorted(main_files, key=lambda x: len(x), reverse=True)[0]

def get_total_paths_over_time(folder_path):
    """Get total paths (leaf nodes) over time using time snapshots"""
    main_file = find_main_file(folder_path)
    if not main_file:
        print(f"Error: Main file not found in {folder_path}")
        return None, None
    
    # Parse main file to get success status
    result_info, _ = parse_main_file_safe_and_success(main_file)
    
    # Find all time snapshot files
    snapshot_pattern = os.path.join(folder_path, '*_t*s.txt')
    files = glob.glob(snapshot_pattern)
    
    if not files:
        print(f"  No time snapshot files found")
        return None, None
    
    # Extract times and sort
    time_data = []
    for filepath in files:
        time_seconds = extract_time_from_filename(filepath)
        if time_seconds is not None:
            time_data.append((time_seconds, filepath))
    
    time_data.sort()
    
    # Track the last-seen state for each prompt
    prompt_history = {}
    
    # For each time point, calculate total paths
    times = [0]
    total_paths_list = [0]
    
    for time_seconds, filepath in time_data:
        # Parse snapshot file
        all_results = parse_tree_file(filepath)
        
        # Update prompt history with current snapshot data
        for result_data in all_results:
            result_num = result_data['result_num']
            leaf_count = count_leaf_nodes(result_data['nodes'])
            prompt_history[result_num] = {
                'paths': leaf_count,
                'last_seen_time': time_seconds
            }
        
        # Calculate total paths from last-seen states of all completed prompts
        total_paths = 0
        for result_num, info in result_info.items():
            # Count this prompt if it was completed by this time
            if info['running_time'] is not None and info['running_time'] <= time_seconds:
                if result_num in prompt_history:
                    total_paths += prompt_history[result_num]['paths']
        
        times.append(time_seconds)
        total_paths_list.append(total_paths)
    
    return times, total_paths_list

def compare_paths():
    """Create a comparison plot of total paths for different settings"""
    print("="*80)
    print("Total Paths Comparison - Over Time")
    print("="*80)
    print(f"\nComparing {len(SETTINGS_FOLDERS)} settings:")
    for i, setting_name in enumerate(SETTINGS_FOLDERS.keys(), 1):
        print(f"  {i}. {setting_name}")
    print()
    
    # Create plot
    plt.figure(figsize=(12, 6.5))
    
    # Set font sizes
    font_size, font_size_legend = 22, 20
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size_legend,
    })
    
    # Plot each setting
    for setting_name, folder_path in SETTINGS_FOLDERS.items():
        print(f"Processing {setting_name}...")
        times, total_paths = get_total_paths_over_time(folder_path)
        
        if times is None or total_paths is None:
            print(f"  Skipping {setting_name} due to errors")
            continue
        
        # Check for times > 1000 seconds
        times_over_1000 = [t for t in times if t > 1000]
        if times_over_1000:
            print(f"  WARNING: Found {len(times_over_1000)} time points > 1000s: {times_over_1000[:5]}...")
            print(f"  Max time: {max(times)}s")
        
        color = SETTINGS_COLORS.get(setting_name, None)
        plt.plot(times, total_paths, '-', linewidth=2.5, label=setting_name, color=color)
        print(f"  Final total paths: {total_paths[-1]} at {times[-1]}s")
    
    plt.xlabel('Search Time Per Query (seconds)')
    plt.ylabel('Total Paths (Leaf Nodes)')
    plt.grid(True)
    plt.legend(loc='upper left')
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Set x-axis to linear scale with range 0-600
    plt.xlim(left=0, right=600)
    
    # Adjust tick label positioning
    plt.tick_params(axis='x', pad=10)
    plt.tick_params(axis='y', pad=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_dir = './figures_mlsys'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'paths_comparison.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"Comparison plot saved to: {os.path.abspath(output_path)}")
    print(f"{'='*80}\n")

def main():
    """Main function - compare total paths for all settings"""
    compare_paths()

if __name__ == "__main__":
    main()

