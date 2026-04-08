#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare success rates across multiple models

Usage:
    python plot_different_models_default.py
    
Configuration:
    Edit MODEL_FOLDERS to define which models to compare
    
Output:
    - Comparison chart saved in ./figures_mlsys/ as model_comparison.pdf
"""

import math
import re as re_module
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Import plot utilities for unified styling
from plot_utils import setup_plot_font_style, get_color

# Set font for better rendering
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# Model name to folder path mapping for comparison
# Use standard model display names (matching parse_results_dashed.py)
MODEL_FOLDERS = {
    "Llama 3.1 (8B)": "/home/shuyi/BOA/results/merged_Llama-3.1-8B-Instruct",
    "Llama 2 (7B)": "/home/shuyi/BOA/results/merge_llama2_7b",
    # "Vicuna 1.5 (7B)": "/home/shuyilin/Jailbreak_oracle/result/20251016_052222_vicuna-7b-v1.5_-1_0.9_0.6_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off",
    "Qwen3 (8B)": "/home/shuyi/BOA/results/merged_qwen3_8b",
    "Gemma 3 (4B)": "/home/shuyi/BOA/results/merged_single_turn_gemma-3-4b-it_hf_p0.95_k64_t1_lh0.0001_cs1_greedy_cacheon_sbufon_jbufon_topkon",
}

# Use unified color scheme from plot_utils
def get_unified_model_color(model_name):
    """Get color using unified color scheme from plot_utils."""
    if "Llama 3.1" in model_name:
        return get_color('models', 'llama31')
    elif "Llama 2" in model_name:
        return get_color('models', 'llama2')
    elif "Vicuna" in model_name:
        return get_color('models', 'vicuna')
    elif "Qwen" in model_name:
        return get_color('models', 'qwen')
    elif "Gemma" in model_name:
        return get_color('models', 'gemma')
    elif "Llama 3" in model_name:  # For Llama 3 (not 3.1)
        return get_color('models', 'llama3')
    else:
        return '#000000'  # Default black color for other models

def get_success_rates_over_time(folder_path):
    """Get success rates over time for a single model using runs.txt"""
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
        safe_match = re_module.search(r'^Safe:\s*(\S+)', block, re_module.MULTILINE)
        jailbreak_found = safe_match is not None and safe_match.group(1) == 'NO'
        time_match = re_module.search(r'^Elapsed time:\s*([\d.]+)s', block, re_module.MULTILINE)
        duration = float(time_match.group(1)) if time_match else 0
        results.append({
            'duration': duration,
            'jailbreak_found': jailbreak_found,
        })

    total_prompts = 128  # Fixed denominator
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

def compare_models():
    """Create a comparison plot of success rates for different models"""
    print("="*80)
    print("Model Comparison - Success Rate Over Time")
    print("="*80)
    print(f"\nComparing {len(MODEL_FOLDERS)} models:")
    for i, model_name in enumerate(MODEL_FOLDERS.keys(), 1):
        print(f"  {i}. {model_name}")
    print()
    
    # Create plot
    plt.figure(figsize=(12, 6.5))
    
    # Set up unified font style
    setup_plot_font_style()
    
    # Plot each model
    for model_name, folder_path in MODEL_FOLDERS.items():
        print(f"Processing {model_name}...")
        times, success_rates = get_success_rates_over_time(folder_path)
        
        if times is None or success_rates is None:
            print(f"  Skipping {model_name} due to errors")
            continue
        
        # Check for times > 1000 seconds
        times_over_1000 = [t for t in times if t > 1000]
        if times_over_1000:
            print(f"  WARNING: Found {len(times_over_1000)} time points > 1000s: {times_over_1000[:5]}...")  # Show first 5
            print(f"  Max time: {max(times)}s")
        
        color = get_unified_model_color(model_name)
        plt.plot(times, success_rates, '-', linewidth=2.5, label=model_name, color=color)
        print(f"  Final success rate: {success_rates[-1]:.2f}% at {times[-1]}s")
    
    
    plt.xlabel('Search Time Per Query (seconds)')
    plt.ylabel('Jailbreak Discovery Rate (%)')
    plt.grid(True)
    plt.legend(loc='upper left')
    
    # Set y-axis to start from 0 and end at 20%
    plt.ylim(bottom=0, top=40)

    # Set x-axis to linear scale with range 0-700
    plt.xlim(left=0, right=600)
    
    # Adjust tick label positioning
    plt.tick_params(axis='x', pad=10)
    plt.tick_params(axis='y', pad=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_dir = './figures_mlsys'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'model_comparison.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"Comparison plot saved to: {os.path.abspath(output_path)}")
    print(f"{'='*80}\n")

def main():
    """Main function - compare all models defined in MODEL_FOLDERS"""
    compare_models()

if __name__ == "__main__":
    main()

