"""
    Helper functions for plot generation.
"""
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List
import matplotlib.font_manager as fm
import matplotlib as mpl

RESULTS_BASE_DIR = "/home/shuyi/Jailbreak_oracle/result"
# RESULTS_BASE_DIR = "./result"


# Change font-family to times-new-roman
# Font-type and figure DPI
mpl.rcParams["figure.dpi"] = 600
# Make text in plots match LaTeX
font_prop = fm.FontProperties(fname='Times New Roman.ttf')
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [font_prop.get_name()] + plt.rcParams['font.serif']


def hex_color(color):
    """Convert matplotlib color to hex."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(color[0]*255), 
        int(color[1]*255), 
        int(color[2]*255)
    )


# Define base colormaps for each category
# Change these to modify the color schemes
EPSILON_COLORMAP = 'Greens_d'
TOPK_COLORMAP    = 'Oranges_d'
TOPP_COLORMAP    = 'PuRd'

# Generate colors for each category
epsilon_colors = sns.color_palette(EPSILON_COLORMAP, n_colors=6)
topp_colors    = sns.color_palette(TOPP_COLORMAP, n_colors=6)
topk_colors    = sns.color_palette(TOPK_COLORMAP, n_colors=3)


# Use seaborn palette for distinct model colors
model_palette = sns.color_palette('Set2', n_colors=5)


COLOR_MAP = {
    'epsilon': {
        1:    hex_color(epsilon_colors[0]),
        1e-1: hex_color(epsilon_colors[1]),
        1e-2: hex_color(epsilon_colors[2]),
        1e-3: hex_color(epsilon_colors[3]),
        1e-4: hex_color(epsilon_colors[4]),
        1e-5: hex_color(epsilon_colors[5]),
    },
    'topk': {
        5:  hex_color(topk_colors[0]),
        10: hex_color(topk_colors[1]),
        20: hex_color(topk_colors[2]),
    },
    'topp': {
        (0.6, 0.9):  hex_color(topp_colors[0]),
        (0.8, 0.6):  hex_color(topp_colors[0]),
        (0.8, 1.0):  hex_color(topp_colors[1]),
        (0.9, 0.6):  hex_color(topp_colors[2]),
        (0.9, 1.0):  hex_color(topp_colors[3]),
        (0.95, 0.6):  hex_color(topp_colors[4]),
        (0.95, 1.0):  hex_color(topp_colors[5]),
        # Additional combinations for plot_compare_combined.py
        (1.0, 0.9):  hex_color(topp_colors[3]),  # Same as (0.9, 1.0)
        (1.0, 0.8):  hex_color(topp_colors[1]),  # Same as (0.8, 1.0)
    },
    'models': {
        'vicuna':  hex_color(model_palette[0]),
        'llama2':  hex_color(model_palette[1]),
        'llama3':  hex_color(model_palette[2]),
        'llama31': hex_color(model_palette[3]),
        'qwen':    hex_color(model_palette[4]),
        'gemma':   hex_color(model_palette[2]),  # Black color for Gemma models
    }
}


def get_color(category: str, value) -> str:
    category_map = COLOR_MAP.get(category, None)
    if category_map is None:
        raise ValueError(f"Unknown category: {category}. Available categories: {list(COLOR_MAP.keys())}")
    color = category_map.get(value, None)
    if color is None:
        raise ValueError(f"Unknown value: {value} for category: {category}. Available values: {list(category_map.keys())}")
    return color


def setup_plot_font_style(font_size=22, font_size_legend=20):
    """Set up unified font style for all plots to ensure consistency."""
    plt.rcParams.update({
        'font.size': font_size,  # Base font size
        'axes.labelsize': font_size,  # Axis labels
        'axes.titlesize': font_size,  # Axis titles
        'xtick.labelsize': font_size,  # X-axis tick labels
        'ytick.labelsize': font_size,  # Y-axis tick labels
        'legend.fontsize': font_size_legend,  # Legend font size
    })


def convert_scientific_notation(text: str) -> str:
    """Convert scientific notation from 1eX format to 10^X format.
    Special case: 1e0 becomes just '1'.
    """
    # Pattern to match 1e followed by optional minus and digits
    pattern = r'1e([+-]?\d+)'
    
    def replace_func(match):
        exponent = match.group(1)
        if exponent == '0' or exponent == '+0' or exponent == '-0':
            return '1'
        else:
            # Remove leading '+' if present
            if exponent.startswith('+'):
                exponent = exponent[1:]
            return f'$10^{{{exponent}}}$'
    
    return re.sub(pattern, replace_func, text)


def analyze_time_series(results: List[Dict]) -> tuple:
    """Analyze number of successes over time for a set of results."""
    all_times = []
    all_successes = []
    
    # Collect all times and successes
    for result in results:
        for prompt_result in result['results']:
            all_times.append(prompt_result['search_time'])
            print(prompt_result['nuanced_judger'])
            all_successes.append(1 if prompt_result['nuanced_judger'] == 'unsafe' else 0)
    
    if not all_times:
        return [], []
    
    # Sort by time
    sorted_indices = np.argsort(all_times)
    sorted_times = np.array(all_times)[sorted_indices]
    sorted_successes = np.array(all_successes)[sorted_indices]
    
    # Calculate number of successes at each time point
    cumulative_successes = np.cumsum(sorted_successes)
    
    return sorted_times, cumulative_successes


def parse_results_file(file_path: str,
                       extract_parameters_from_filename) -> Dict:
    results = []

    with open(file_path, 'r') as f:
        content = f.read()
        
    # Extract parameters from filename
    params = extract_parameters_from_filename(file_path)
    
    # Split content by prompt sections
    prompt_sections = content.split('###Prompt')
    print(len(prompt_sections))

    
    # Only process first 60 prompts
    cal = 0
    for section in prompt_sections[1:]:  # Skip the first empty split
        try:
            # Extract prompt number
            prompt_num_str = section.split('###')[0].strip()
            if not prompt_num_str:  # Skip if empty
                continue
            prompt_num = int(prompt_num_str)
            
            # Extract probs
            probs_match = re.search(r'###Probs###:\s*([-\d.]+)', section)
            probs = float(probs_match.group(1)) if probs_match else None
            
            # Extract search_time
            search_time_match = re.search(r'search_time:\s*([\d.]+)', section)
            search_time = float(search_time_match.group(1)) if search_time_match else None
            
            # Extract nuanced_judger
            nuanced_judger_match = re.search(r'nuanced_judger:\s*(\w+)', section)
            nuanced_judger = nuanced_judger_match.group(1) if nuanced_judger_match else None

            if nuanced_judger == None:
                continue
            else: 
                cal+=1
            # if cal > 62:
            #     break

            
            # Extract success
            success_match = re.search(r'success:\s*(\w+)', section)
            success = success_match.group(1) if success_match else None

            
            results.append({
                'prompt_num': prompt_num,
                'probs': probs,
                'search_time': search_time,
                'nuanced_judger': nuanced_judger,
                'success': success
            })
            
        except Exception as e:
            print(f"Error processing section in {file_path}: {e}")
            continue
    
    return {
        'file_path': file_path,
        'model_name': params['model_name'],
        'topk': params['topk'],
        'temperature': params['temperature'],
        'topp': params['topp'],
        'results': results
    }


def process_files(file_paths: List[str],
                  experiment_type: str,
                  extract_parameters_from_filename) -> Dict:
    """Process multiple files and organize results by model and experiment parameter."""
    all_results = {}
    
    print(f"\nProcessing files for experiment type: {experiment_type}")
    print(f"Number of files to process: {len(file_paths)}")
    
    for file_path in file_paths:
        try:
            print(f"Processing file: {file_path}")
            result = parse_results_file(file_path, extract_parameters_from_filename)
            model_name = result['model_name']
            if model_name not in all_results:
                all_results[model_name] = {}
            
            # Extract the relevant parameter based on experiment type
            if experiment_type == 'topk':
                param_value = result['topk']
            elif experiment_type == 'topp':
                # Create a tuple of (topp, temperature) as the key
                param_value = (result['topp'], result['temperature'])
            elif experiment_type == 'temperature':
                param_value = result['temperature']
            else:  # all_default case
                param_value = model_name
            
            if param_value not in all_results[model_name]:
                all_results[model_name][param_value] = []
            all_results[model_name][param_value].append(result)
            print(f"Successfully processed file for model: {model_name}, param: {param_value}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    print(f"\nProcessed results summary:")
    for model_name, param_results in all_results.items():
        print(f"Model: {model_name}")
        for param_value, results in param_results.items():
            print(f"  Parameter: {param_value}, Number of results: {len(results)}")
    
    return all_results

