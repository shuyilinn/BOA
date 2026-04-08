#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual tree side-by-side visualization script - display two prompt trees side by side

Usage:
1. Modify the configuration parameters below
2. Run: python dual_tree_viz.py
"""

import os
import matplotlib.pyplot as plt
from tree_visualizer import parse_tree_file, build_tree_structure, find_jailbreak_paths, STATUS_COLORS
from tree_visualizer import _layout_radial_grouped, _ordered_children
import numpy as np
import matplotlib.patches as patches
import matplotlib

# Import plot utilities for unified styling
from plot_utils import setup_plot_font_style

# Set font for better rendering (consistent with plot_compare_topp.py)
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def draw_single_tree(ax, root, tree, result_num, title="", base_step=3.0):
    """Draw a single tree on the specified axes"""
    root_key = root['line_num']
    
    # Use grouped sector layout
    positions = _layout_radial_grouped(
        tree,
        root_key,
        base_step=base_step,
        angle_start=0.0,
        angle_end=2*np.pi,
    )
    
    # Find edges on jailbreak paths
    jailbreak_edges = find_jailbreak_paths(tree, root)

    # Edges - parent-child connection lines
    for u, data in tree.items():
        if u not in positions:
            continue
        x, y = positions[u]
        u_depth = tree[u]['node']['depth']
        for v in data['children']:
            if v not in positions:
                continue
            xv, yv = positions[v]
            
            if (u, v) in jailbreak_edges:
                line_color = 'red'
                alpha = 0.9
                lw = 1.5
                z = 3
            else:
                line_color = 'gray'
                alpha = 0.6
                lw = 0.6
                z = 1
            
            ax.plot([x, xv], [y, yv],
                    color=line_color,
                    alpha=alpha,
                    linewidth=lw,
                    zorder=z)
    
    # Nodes
    for nid, data in tree.items():
        if nid not in positions:
            continue
        node = data['node']
        x, y = positions[nid]
        
        # Draw node circles
        if nid == root_key:
            radius = 1.0  # Root node
            node_color = STATUS_COLORS['ROOT']
        else:
            radius = 0.7  # Regular node
            node_color = node['color']
            
        circle = plt.Circle((x, y), radius, facecolor=node_color, 
                           edgecolor='black', linewidth=0, alpha=1.0, zorder=3)
        ax.add_patch(circle)
    
    # Set axes
    ax.axis('off')
    
    # Boundary settings
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    
    if xs and ys:  # Ensure there are nodes
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        pad_x = (xmax - xmin) * 0.05 + 0.1
        pad_y = (ymax - ymin) * 0.05 + 0.1
        
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect('equal', adjustable='datalim')
    
    # Add title
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    
    return ax

def create_dual_tree_visualization(file_name, prompt1, prompt2, output_path, max_depth=None):
    """
    Create dual tree side-by-side visualization

    Args:
        file_name: Input file path
        prompt1: First prompt number
        prompt2: Second prompt number
        output_path: Output file path
        max_depth: Maximum depth limit
    """
    print(f"Dual tree side-by-side visualization tool")
    print(f"Input file: {os.path.basename(file_name)}")
    print(f"Prompt 1: {prompt1}")
    print(f"Prompt 2: {prompt2}")
    if max_depth:
        print(f"Max depth: {max_depth}")
    print("=" * 60)
    
    # Parse data for both prompts
    results1 = parse_tree_file(file_name, prompt1)
    results2 = parse_tree_file(file_name, prompt2)
    
    if not results1:
        print(f"Error: Cannot find Prompt {prompt1}")
        return False
    if not results2:
        print(f"Error: Cannot find Prompt {prompt2}")
        return False

    # Build tree structure
    root1, tree1 = build_tree_structure(results1[0]['nodes'])
    root2, tree2 = build_tree_structure(results2[0]['nodes'])
    
    if root1 is None or root2 is None:
        print("Error: Cannot build tree structure")
        return False
    
    # Apply depth filtering
    if max_depth is not None:
        from tree_visualizer import filter_tree_by_depth
        tree1 = filter_tree_by_depth(tree1, root1, max_depth)
        tree2 = filter_tree_by_depth(tree2, root2, max_depth)
    
    # Set unified font style
    setup_plot_font_style()
    
    # Create figure and subplots - maximize compactness
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9.5))
    
    # Draw both trees
    print(f"Drawing Prompt {prompt1}...")
    draw_single_tree(ax1, root1, tree1, prompt1, "")  # No subtitle displayed

    print(f"Drawing Prompt {prompt2}...")
    draw_single_tree(ax2, root2, tree2, prompt2, "")  # No subtitle displayed

    # No main title
    # base_name = os.path.splitext(os.path.basename(file_name))[0]
    # fig.suptitle(f'Tree Comparison: {base_name}', fontsize=20, y=0.95)
    
    # Add legend (only once, placed on the right side)
    existing_statuses = set()
    for node_data in tree1.values():
        existing_statuses.add(node_data['node']['status'])
    for node_data in tree2.values():
        existing_statuses.add(node_data['node']['status'])
    
    legend_elements = []
    legend_elements.append(patches.Patch(color=STATUS_COLORS['ROOT'], label='Root'))
    
    for status in sorted(existing_statuses):
        if status in STATUS_COLORS:
            color = STATUS_COLORS[status]
            label = status.replace('_', ' ').title()
            legend_elements.append(patches.Patch(color=color, label=label))
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper center', fontsize=20, 
                  bbox_to_anchor=(0.5, 1), ncol=len(legend_elements))
    
    # Manually set axes positions to fill the entire canvas
    ax1.set_position([0.0, -0.1, 0.53, 1.15])  # [left, bottom, width, height]
    ax2.set_position([0.48, -0.1, 0.53, 1.15])

    # Add (a) and (b) labels below the subplots
    # Calculate label position: below the bottom of subplots
    fig.text(0.265, 0.0, '(a)', fontsize=24, ha='center', va='bottom', fontweight='bold')
    fig.text(0.735, 0.0, '(b)', fontsize=24, ha='center', va='bottom', fontweight='bold')

    # Trim edges completely when saving
    plt.savefig(output_path, format='pdf',bbox_inches=None)
    print(f"Dual tree visualization saved to: {output_path}")
    
    plt.close()
    return True

# ===== Configuration Parameters =====
file_name = "/home/shuyilin/Jailbreak_oracle/result/20251018_033631_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off/20251018_033631_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off.txt"

# Two prompts to compare
prompt1 = 3  # First prompt
prompt2 = 7  # Second prompt

# Output file path
output_path = "dual_tree_comparison.pdf"

# Maximum depth limit (None means no limit)
max_depth = None  # Or set to a specific number, e.g., max_depth = 5

# ===== Start dual tree visualization =====
if __name__ == "__main__":
    print("Dual tree side-by-side visualization tool")
    print("=" * 40)
    
    success = create_dual_tree_visualization(file_name, prompt1, prompt2, output_path, max_depth)
    
    if success:
        print("\nDual tree visualization complete!")
    else:
        print("\nDual tree visualization failed!")
