#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch dual tree side-by-side visualization script - generate multiple dual tree comparison plots

Usage:
1. Modify the configuration parameters below
2. Run: python batch_dual_tree_viz.py
"""

import os
from dual_tree_viz import create_dual_tree_visualization

def batch_dual_visualize(file_name, prompt_pairs, output_dir, max_depth=None):
    """
    Batch generate multiple dual tree comparison plots

    Args:
        file_name: Input file path
        prompt_pairs: List of prompt comparison pairs, each element is a (prompt1, prompt2) tuple
        output_dir: Output directory
        max_depth: Maximum depth limit
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Batch dual tree side-by-side visualization tool")
    print(f"Input file: {os.path.basename(file_name)}")
    print(f"Output directory: {output_dir}")
    print(f"Number of comparison pairs: {len(prompt_pairs)}")
    if max_depth:
        print(f"Max depth: {max_depth}")
    print("=" * 60)
    
    success_count = 0
    total_count = len(prompt_pairs)
    
    for i, (prompt1, prompt2) in enumerate(prompt_pairs, 1):
        print(f"\n[{i}/{total_count}] Processing comparison: Prompt {prompt1} vs Prompt {prompt2}...")

        # Generate output file name
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        output_name = f"dual_tree_{base_name}_p{prompt1}_vs_p{prompt2}.pdf"
        output_path = os.path.join(output_dir, output_name)
        
        try:
            success = create_dual_tree_visualization(
                file_name=file_name,
                prompt1=prompt1,
                prompt2=prompt2,
                output_path=output_path,
                max_depth=max_depth
            )
            
            if success:
                print(f"  Success: {output_name}")
                success_count += 1
            else:
                print(f"  Failed: {output_name}")
                
        except Exception as e:
            print(f"  Error: {output_name} - {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{total_count} comparison plots generated successfully")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return success_count == total_count

# ===== Configuration Parameters =====
file_name = "/home/shuyilin/Jailbreak_oracle/result/20251018_033631_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off/20251018_033631_Llama-3.1-8B-Instruct_-1_0.6_0.9_0.0001_atkSamp=on-unif=1.00-cache=on-sbuf=on-jbuf=on-block=off.txt"

# List of prompt pairs to compare
prompt_pairs = [
    (1, 7),   # Prompt 1 vs Prompt 2
    (3, 7), 

]

# Output directory
output_dir = "dual_tree_visualizations"

# Maximum depth limit (None means no limit)
max_depth = None  # Or set to a specific number, e.g., max_depth = 5

# ===== Start batch dual tree visualization =====
if __name__ == "__main__":
    print("Batch dual tree side-by-side visualization tool")
    print("=" * 40)
    
    success = batch_dual_visualize(file_name, prompt_pairs, output_dir, max_depth)
    
    if success:
        print("\nAll dual tree comparison plots generated successfully!")
    else:
        print("\nSome comparison plots failed to generate. Please check the error messages.")
