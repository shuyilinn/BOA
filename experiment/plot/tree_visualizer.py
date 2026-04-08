#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree Visualizer - Visualize tree structures with different colors for different node states

Features:
1. Parse tree structure files
2. Create tree visualization
3. Use different colors for different node states
4. Support saving as image files
5. Support batch processing of multiple files

Usage:
    # Process a single file
    python tree_visualizer.py <main_file_path> [result_number]

    # Process a file list
    python tree_visualizer.py file_list.txt --list [result_number]

    # Process all txt files in a directory
    python tree_visualizer.py <directory_path> --dir [result_number]

    # Save image
    python tree_visualizer.py <input> --save output_prefix

    # Set maximum depth
    python tree_visualizer.py <input> --max-depth 10

    # Set node count threshold
    python tree_visualizer.py <input> --max-nodes 500
"""

import re
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import defaultdict
import argparse

# Node status color mapping - matching the target image color scheme
STATUS_COLORS = {
    'COMPLETED': '#32CD32',    # Forest green - completed paths (slightly darker green)
    'JAILBREAK': '#FF4C4C',    # Red - jailbreak paths
    'EVALUATED': '#6BB8FF',    # Sky blue - evaluated but not completed
    'QUEUED': '#FFD700',       # Gold - queued nodes
    'CUT': '#FFA500',          # Orange - pruned nodes
    'ROOT': '#8B4513'          # Brown - root node
}

# Normalize various log TAG formats to color dictionary keys
STATUS_ALIASES = {
    "COMPLETED": "COMPLETED",
    "SUCCESS": "COMPLETED",
    "JAILBREAK": "JAILBREAK",
    "EVALUATED": "EVALUATED",
    "CREATED": "EVALUATED",
    "QUEUED": "QUEUED",
    "CUT": "CUT",
    "CUT/PRUNED": "CUT",
    "PRUNED": "CUT"
}

def normalize_status(raw_tag: str) -> str:
    if not raw_tag:
        return "QUEUED"  # Empty status defaults to queued
    tag = raw_tag.upper().strip()
    # Remove emojis and surrounding whitespace
    tag = tag.replace("🔓", "").replace("✂️", "").strip()
    # Support CUT/PRUNED style slash notation
    if tag in STATUS_ALIASES:
        return STATUS_ALIASES[tag]
    # Try substring matching
    for key in STATUS_ALIASES:
        if key in tag:
            return STATUS_ALIASES[key]
    return "QUEUED"  # Unrecognized status defaults to queued

def parse_tree_file(filepath, target_result=None):
    """Parse tree structure file and extract tree structure for specified result"""
    results = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find all Result delimiters
    result_indices = []
    for i, line in enumerate(lines):
        if line.strip().startswith('==================== Result'):
            result_num_match = re.search(r'Result (\d+)', line)
            if result_num_match:
                result_num = int(result_num_match.group(1))
                result_indices.append((i, result_num))
    
    # If a specific result is specified, only process that result
    if target_result is not None:
        result_indices = [(idx, num) for idx, num in result_indices if num == target_result]
        if not result_indices:
            print(f"Error: Result {target_result} not found in file")
            return []
    
    # Parse each Result
    for idx, (start_idx, result_num) in enumerate(result_indices):
        # Determine the end position of the current Result
        if idx < len(result_indices) - 1:
            end_idx = result_indices[idx + 1][0]
        else:
            # If this is the last Result, find the start of the next Result
            # Or if this is the only Result, process to end of file
            if len(result_indices) == 1:
                # Only one Result, need to find the start of the next Result
                next_result_start = None
                for i in range(start_idx + 1, len(lines)):
                    if lines[i].strip().startswith('==================== Result'):
                        next_result_start = i
                        break
                end_idx = next_result_start if next_result_start else len(lines)
            else:
                end_idx = len(lines)
        
        print(f"DEBUG: Processing Result {result_num}, lines {start_idx} to {end_idx} (total: {end_idx - start_idx} lines)")
        
        # Find the Tree section
        tree_start = None
        for i in range(start_idx, end_idx):
            line = lines[i]
            if line.strip().startswith('Tree:'):
                tree_start = i + 1
                break
        
        if tree_start is None:
            print(f"Warning: Result {result_num} has no Tree section")
            continue
        
        # Parse tree nodes (modified pattern to match -- or - prefix, and status with special characters)
        nodes = []
        # -+ matches one or more dashes (-- or -)
        # [^\]]+ matches any character inside brackets (including emojis and spaces)
        # Supports tokens enclosed in single or double quotes
        pattern = r"-+\s*\[([^\]]+)\] (?:'([^']*)'|\"([^\"]*)\") \(id: (\d+)\), logP: ([-\d.]+), score: ([-\d.]+)"
        
        debug_first_lines = 0
        for i in range(tree_start, end_idx):
            line = lines[i]
            if line.strip().startswith('===') or not line.strip():
                continue
            
            # Calculate indent level (depth = indent spaces / 2)
            # Note: -- prefix is root node (depth=0), - prefix is child node (depth=1)
            indent = (len(line) - len(line.lstrip())) // 2
            depth = indent
            
            # Debug: print first few lines
            if debug_first_lines < 3:
                print(f"DEBUG parse line {i}: depth={depth}, line='{line.rstrip()}'")
                debug_first_lines += 1
            
            match = re.search(pattern, line)
            if debug_first_lines <= 3:
                if match:
                    print(f"  -> Matched! status={match.group(1)}, token={match.group(2)}, id={match.group(3)}")
                else:
                    print(f"  -> NOT matched!")
            
            if match:
                groups = match.groups()
                raw_status = groups[0]
                # Handle single and double quote cases
                token = groups[1] if groups[1] is not None else groups[2]  # groups[1] is single quote, groups[2] is double quote
                node_id = groups[3]
                log_p = groups[4]
                score = groups[5]
                
                # Use the new status normalization function
                status = normalize_status(raw_status)
                color = STATUS_COLORS.get(status, STATUS_COLORS['QUEUED'])
                
                nodes.append({
                    'status': status,
                    'token': token,
                    'id': int(node_id),
                    'logP': float(log_p),
                    'score': float(score),
                    'depth': depth,
                    'line_num': i,
                    'color': color
                })
        
        if nodes:
            print(f"DEBUG: Result {result_num} parsed {len(nodes)} nodes")
            results.append({
                'result_num': result_num,
                'nodes': nodes
            })
        else:
            print(f"DEBUG: Result {result_num} has no nodes")
    
    return results

def build_tree_structure(nodes):
    """Build tree structure and establish parent-child relationships
    Note: Uses line_num as the unique identifier for nodes, since the same token id may appear multiple times
    """
    tree = {}
    root = None
    
    # Add depth cache at the beginning of function
    last_at_depth = {}
    
    # Find root node (first node with depth=0)
    for node in nodes:
        if node['depth'] == 0:
            root = node
            print(f"DEBUG build_tree: Found root node line={node['line_num']}, id={node['id']}, depth={node['depth']}, token='{node['token']}'")
            break
    
    if root is None:
        print(f"DEBUG build_tree: No root node found! Total nodes: {len(nodes)}")
        # Show depth of first 5 nodes
        for i, n in enumerate(nodes[:5]):
            print(f"  Node {i}: id={n['id']}, depth={n['depth']}, token='{n['token']}'")
        return None, {}
    
    # Build tree structure using line_num as unique key
    tree[root['line_num']] = {
        'node': root,
        'children': []
    }
    last_at_depth[root['depth']] = root['line_num']
    
    # Find parent node for each node
    for i, node in enumerate(nodes):
        if node['depth'] == 0:
            continue
        
        # Update depth cache
        last_at_depth[node['depth']] = node['line_num']
        
        # Prefer cache when finding parent
        parent = None
        for d in range(node['depth'] - 1, -1, -1):
            if d in last_at_depth:
                parent_line = last_at_depth[d]
                parent = next((n for n in nodes if n['line_num'] == parent_line), None)
                break
        
        # If cache method fails, fall back to the original method
        if parent is None:
            for j in range(i-1, -1, -1):
                if nodes[j]['depth'] < node['depth']:
                    parent = nodes[j]
                    break
        
        if parent:
            parent_key = parent['line_num']
            node_key = node['line_num']
            
            if parent_key not in tree:
                tree[parent_key] = {
                    'node': parent,
                    'children': []
                }
            # Avoid adding the same child twice
            if node_key not in tree[parent_key]['children']:
                tree[parent_key]['children'].append(node_key)
            # Only add when node_key is not in tree, to avoid overwriting existing nodes (especially root)
            if node_key not in tree:
                tree[node_key] = {
                    'node': node,
                    'children': []
                }
    
    # Debug: check if root node is in tree
    root_key = root['line_num']
    print(f"DEBUG build_tree: Built tree with {len(tree)} nodes from {len(nodes)} input nodes")
    print(f"DEBUG build_tree: Root line={root_key}, id={root['id']} in tree? {root_key in tree}")
    if root_key in tree:
        print(f"  Root node depth in tree: {tree[root_key]['node']['depth']}")
    
    # Check for duplicate node IDs (this is normal, same token can appear multiple times)
    node_ids = [n['id'] for n in nodes]
    unique_ids = set(node_ids)
    if len(node_ids) != len(unique_ids):
        print(f"INFO: Same token ID appears multiple times - Total nodes: {len(node_ids)}, Unique token IDs: {len(unique_ids)}")
        # Find duplicate IDs
        from collections import Counter
        id_counts = Counter(node_ids)
        duplicates = {nid: count for nid, count in id_counts.items() if count > 1}
        print(f"  Most frequent IDs (first 3): {list(duplicates.items())[:3]}")
    
    return root, tree

def _ordered_children(tree, node_id):
    """Return children in order of appearance (line_num) to ensure stable layout"""
    if node_id not in tree:
        print(f"WARNING: node_id {node_id} not in tree!")
        return []
    ch = tree[node_id]['children']
    # Filter out children not in tree (defensive programming)
    valid_children = [cid for cid in ch if cid in tree]
    if len(valid_children) < len(ch):
        print(f"WARNING: Node {node_id} has {len(ch)-len(valid_children)} children not in tree")
    return sorted(valid_children, key=lambda cid: tree[cid]['node']['line_num'])

def _subtree_size(tree, node_id, _cache):
    """Return the subtree size of this node (including itself).
    Used to allocate angle bandwidth proportionally.
    """
    if node_id in _cache:
        return _cache[node_id]
    children = _ordered_children(tree, node_id)
    if not children:
        _cache[node_id] = 1
        return 1
    total = 1
    for c in children:
        total += _subtree_size(tree, c, _cache)
    _cache[node_id] = total
    return total

def _layout_radial_grouped(
    tree,
    root_id,
    base_step=6.0,
    angle_start=0.0,
    angle_end=2*np.pi,
):
    """
    Layered radius remains constant: radius = depth * base_step
    But angle is no longer evenly distributed across the full circle; instead each parent node
    occupies a contiguous angle sector, and its children and descendants live within that sector.

    Returns positions: {node_id: (x, y)}
    """
    positions = {}
    depth_cache = {nid: data['node']['depth'] for nid, data in tree.items()}
    subtree_cache = {}

    # Pre-compute subtree size for each node, used for angle allocation
    _subtree_size(tree, root_id, subtree_cache)

    def assign_positions(node_id, theta0, theta1):
        """
        Place node_id at its proper radius, then split its children's angle range by subtree size.
        theta0, theta1 are the dedicated angle sector for this node_id.
        """
        d = depth_cache[node_id]
        r = (d + 0.5) * base_step  # depth=0 -> r=0.5*base_step, reduce center whitespace
        
        # Push leaf nodes slightly outward for clearer edges
        if not _ordered_children(tree, node_id):
            r += base_step * 0.15  # Reduce extra offset for leaf nodes

        # Use the midpoint of the sector for the angle
        theta_center = 0.5 * (theta0 + theta1)

        x = r * np.cos(theta_center)
        y = r * np.sin(theta_center)
        positions[node_id] = (x, y)

        children = _ordered_children(tree, node_id)
        if not children:
            return

        # Calculate total subtree size of all children
        sizes = [ _subtree_size(tree, c, subtree_cache) for c in children ]
        total = sum(sizes)

        # Split [theta0, theta1] angle range into len(children) segments,
        # each segment width proportional to the child's subtree size
        cur_theta = theta0
        for c, sz in zip(children, sizes):
            span = (theta1 - theta0) * (sz / total)
            child_theta0 = cur_theta
            child_theta1 = cur_theta + span
            cur_theta += span
            assign_positions(c, child_theta0, child_theta1)

    assign_positions(root_id, angle_start, angle_end)
    return positions

def _layout_tidy(tree, root_id, level_gap=1.6, sep=1.0):
    """
    Simplified Reingold-Tilford:
      - Leaves: placed sequentially (x accumulates)
      - Internal nodes: x is the average of all children's x
      - y = -depth * level_gap
    Returns positions: {node_id: (x, y)}
    """
    positions = {}
    next_x = [0.0]  # Use list for mutable closure

    def dfs(u):
        children = _ordered_children(tree, u)
        if not children:                 # Leaf
            x = next_x[0]
            next_x[0] += sep
        else:
            xs = []
            for v in children:
                xv = dfs(v)
                xs.append(xv)
            x = sum(xs) / len(xs)        # Center parent node
        y = -tree[u]['node']['depth'] * level_gap
        positions[u] = (x, y)
        return x

    dfs(root_id)
    return positions

def find_jailbreak_paths(tree, root):
    """Find all edges on paths from root to JAILBREAK nodes
    Note: tree keys are line_num, not token id
    """
    jailbreak_edges = set()
    
    # Find all JAILBREAK nodes (key is line_num)
    jailbreak_nodes = [line_num for line_num, data in tree.items() 
                      if data['node']['status'] == 'JAILBREAK']
    
    print(f"DEBUG: Found {len(jailbreak_nodes)} JAILBREAK nodes")
    
    # Build parent mapping (key is line_num)
    parent_map = {}
    for node_line, node_data in tree.items():
        for child_line in node_data['children']:
            if child_line in parent_map:
                print(f"WARNING: Node line={child_line} has multiple parents! Old: {parent_map[child_line]}, New: {node_line}")
            parent_map[child_line] = node_line
    
    root_key = root['line_num']
    print(f"DEBUG: Built parent_map with {len(parent_map)} entries")
    print(f"DEBUG: Root line={root_key}, Root in parent_map? {root_key in parent_map}")
    
    # For each JAILBREAK node, trace back to root
    for jailbreak_line in jailbreak_nodes[:2]:  # Only trace first 2 for debugging
        print(f"\nDEBUG: Tracing path for JAILBREAK node line={jailbreak_line}:")
        current = jailbreak_line
        path = [current]
        steps = 0
        while current in parent_map and steps < 50:  # Limit steps to prevent infinite loop
            parent = parent_map[current]
            # Record edge (parent -> child)
            jailbreak_edges.add((parent, current))
            path.append(parent)
            if steps < 10:  # Only print first 10 steps
                print(f"  line {current} <- line {parent}")
            current = parent
            steps += 1
        
        if steps >= 50:
            print(f"  WARNING: Stopped after {steps} steps (possible loop)")
        elif current == root_key:
            print(f"  SUCCESS: Reached root at line {root_key}! Total steps: {steps}")
        else:
            print(f"  WARNING: Stopped at line {current}, not root {root_key}. Steps: {steps}")
    
    # Process all JAILBREAK nodes (without printing)
    for jailbreak_line in jailbreak_nodes:
        current = jailbreak_line
        steps = 0
        while current in parent_map and steps < 100:  # Prevent infinite loop
            parent = parent_map[current]
            jailbreak_edges.add((parent, current))
            current = parent
            steps += 1
    
    print(f"DEBUG: Found {len(jailbreak_edges)} jailbreak edges")
    
    return jailbreak_edges

def deduplicate_tree_by_depth(tree, root):
    """Deduplicate nodes with the same (token, status) at the same depth

    Merge strategy:
    - Nodes with same depth, token, and status are merged into one
    - The merged node inherits all children from the original nodes
    - Uses the earliest line_num as the key for the merged node
    """
    print("\n=== Deduplicating nodes by depth ===")
    
    # Group nodes by depth
    depth_groups = {}
    for node_key, node_data in tree.items():
        depth = node_data['node']['depth']
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append((node_key, node_data))
    
    # Build new tree
    new_tree = {}
    old_to_new_mapping = {}  # old key -> new key mapping
    
    for depth in sorted(depth_groups.keys()):
        nodes = depth_groups[depth]
        
        # Group by (token, status)
        signature_groups = {}
        for node_key, node_data in nodes:
            node = node_data['node']
            signature = (node['token'], node['status'])
            
            if signature not in signature_groups:
                signature_groups[signature] = []
            signature_groups[signature].append((node_key, node_data))
        
        # For each signature group, keep the first node and map others to it
        for signature, group in signature_groups.items():
            # Use first node as representative
            representative_key, representative_data = group[0]
            
            # Collect all children (deduplicated)
            all_children = set()
            for node_key, node_data in group:
                all_children.update(node_data['children'])
                # Record mapping
                old_to_new_mapping[node_key] = representative_key
            
            # Create new node
            new_tree[representative_key] = {
                'node': representative_data['node'],
                'children': list(all_children)
            }
            
            if len(group) > 1:
                token_display = representative_data['node']['token'][:10] if representative_data['node']['token'] else ''
                print(f"  Depth {depth}: Merged {len(group)} nodes with token='{token_display}', status={signature[1]}")
    
    # Update children references: map old keys to new keys
    for node_key in new_tree:
        new_tree[node_key]['children'] = [
            old_to_new_mapping.get(child_key, child_key) 
            for child_key in new_tree[node_key]['children']
            if old_to_new_mapping.get(child_key, child_key) in new_tree
        ]
        # Deduplicate children
        new_tree[node_key]['children'] = list(set(new_tree[node_key]['children']))
    
    print(f"Original tree: {len(tree)} nodes -> Deduplicated tree: {len(new_tree)} nodes")
    print("="*30 + "\n")
    
    return new_tree

def print_tree_structure(tree, root):
    """Print tree structure info: count nodes by depth"""
    print("\n=== Tree Structure Summary ===")
    print(f"Total nodes in tree: {len(tree)}")
    
    # Count by depth
    depth_stats = {}
    status_stats = {}
    
    for node_key, node_data in tree.items():
        node = node_data['node']
        depth = node['depth']
        status = node['status']
        
        if depth not in depth_stats:
            depth_stats[depth] = {'count': 0, 'nodes': []}
        depth_stats[depth]['count'] += 1
        depth_stats[depth]['nodes'].append((node_key, node['token'], node['id'], status))
        
        status_stats[status] = status_stats.get(status, 0) + 1
    
    # Print depth statistics
    print("\nNodes by depth:")
    for depth in sorted(depth_stats.keys()):
        info = depth_stats[depth]
        print(f"  Depth {depth}: {info['count']} nodes")
        # Show first 3 nodes as examples
        for i, (line, token, tid, status) in enumerate(info['nodes'][:3]):
            token_display = token[:10] if token else ''
            print(f"    - line={line}, id={tid}, token='{token_display}', status={status}")
        if len(info['nodes']) > 3:
            print(f"    ... and {len(info['nodes'])-3} more")
    
    # Print status statistics
    print("\nNodes by status:")
    for status, count in sorted(status_stats.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count} nodes")
    
    print("="*30 + "\n")

def draw_tree(root, tree, result_num, save_path=None, force_simplified=False, max_depth=None, output_format='png'):
    """Draw tree visualization (radial layout)"""
    total_nodes = len(tree)
    
    if total_nodes == 0:
        print(f"Error: Tree is empty, nothing to draw!")
        return
    
    root_key = root['line_num']
    if root_key not in tree:
        print(f"Error: Root node (line={root_key}, id={root['id']}) not in filtered tree!")
        return
    
    # No deduplication, each line_num node remains independent
    total_nodes = len(tree)
    
    # Optional: print structure info (useful for debugging depth/status distribution)
    print_tree_structure(tree, root)
    
    # Removed max node count limit, use full visualization directly

    # ======= Use grouped sector layout: nodes in the same subtree cluster together =======
    positions = _layout_radial_grouped(
        tree,
        root_key,
        base_step=3.0,        # Reduce radius step for tighter center
        angle_start=0.0,
        angle_end=2*np.pi,    # Root node takes the full circle by default
    )

    figsize = (16, 16)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Find edges on jailbreak paths
    jailbreak_edges = find_jailbreak_paths(tree, root)

    # Edges - parent-child connection lines (with depth attenuation effect)
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
                lw = 2.0
                z = 3
            else:
                line_color = 'gray'
                # Greater depth = thinner and more transparent lines
                # alpha = max(0.15, 0.6 - 0.03 * u_depth)
                # lw = max(0.3, 1.2 - 0.05 * u_depth)
                alpha  =0.6
                lw = 1.2
                z = 1

            ax.plot([x, xv], [y, yv],
                    color=line_color,
                    alpha=alpha,
                    linewidth=lw,
                    zorder=z)

    # Nodes (scaled down appropriately to avoid text crowding)
    from matplotlib.patches import FancyBboxPatch
    for nid, data in tree.items():
        if nid not in positions:
            continue
        node = data['node']
        x, y = positions[nid]

        token = node['token'] or ''
        token_display = token if len(token) <= 10 else token[:10] + '…'
        if token_display in ('<|eot_id|>', '<|eos|>'):
            token_display = 'EOS'

        # Draw node circles
        # Root node larger and more prominent (fix: use line_num instead of id)
        if nid == root_key:
            radius = 1  # Larger root node
            edge_width = 0
            node_color = STATUS_COLORS['ROOT']  # Root node uses brown
        else:
            radius = 0.5  # Regular node
            edge_width = 0
            node_color = node['color']
            
        circle = plt.Circle((x, y), radius, facecolor=node_color, 
                           edgecolor='black', linewidth=edge_width, alpha=1.0, zorder=3)
        ax.add_patch(circle)

        # No text labels (graphics only display)

    ax.axis('off')

    # Bounds - simply enclose all nodes
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    pad_x = (xmax - xmin) * 0.005 + 0.05
    pad_y = (ymax - ymin) * 0.01 + 0.1 
    
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    
    ax.set_aspect('equal', adjustable='datalim')

    # Add legend
    existing_statuses = set()
    for node_data in tree.values():
        existing_statuses.add(node_data['node']['status'])
    
    legend_elements = []
    
    # First add root node legend entry
    legend_elements.append(patches.Patch(color=STATUS_COLORS['ROOT'], label='Root'))
    
    # Then add other statuses
    for status in sorted(existing_statuses):
        if status in STATUS_COLORS:
            color = STATUS_COLORS[status]
            label = status.replace('_', ' ').title()
            legend_elements.append(patches.Patch(color=color, label=label))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=20, 
                  frameon=True, fancybox=True, shadow=True, 
                  bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout(pad=0.1)
    if save_path:
        # PDF format doesn't need dpi parameter, use tighter margins
        if output_format == 'pdf':
            plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        print(f"Tree visualization saved to: {save_path}")

def draw_simplified_tree(root, tree, result_num, save_path=None, output_format='png'):
    """Draw simplified tree visualization for large tree structures"""
    print(f"Warning: Tree has too many nodes. Please use --max-depth to limit the visualization.")
    return None

def filter_tree_by_depth(tree, root, max_depth):
    """Filter tree, keeping only nodes within the specified depth"""
    if max_depth is None:
        return tree
    
    filtered_tree = {}
    for node_id, node_data in tree.items():
        node = node_data['node']
        if node['depth'] <= max_depth:
            # Only keep child nodes within the depth range
            filtered_children = [
                child_id for child_id in node_data['children']
                if child_id in tree and tree[child_id]['node']['depth'] <= max_depth
            ]
            filtered_tree[node_id] = {
                'node': node,
                'children': filtered_children
            }
    return filtered_tree

def process_single_file(filepath, result_num, save_path=None, max_depth=None, output_format='png'):
    """Process a single file"""
    if not os.path.exists(filepath):
        print(f"Error: File does not exist: {filepath}")
        return False
    
    # Parse file
    if result_num == 'all':
        results = parse_tree_file(filepath)
    else:
        try:
            result_num_int = int(result_num)
            results = parse_tree_file(filepath, result_num_int)
        except ValueError:
            print("Error: result_num must be a number or 'all'")
            return False
    
    if not results:
        print(f"No valid results found in {filepath}")
        return False
    
    # Visualize each result
    success_count = 0
    for result in results:
        result_num_actual = result['result_num']
        nodes = result['nodes']
        
        print(f"\nProcessing {os.path.basename(filepath)} - Result {result_num_actual} with {len(nodes)} nodes...")
        
        # Build tree structure
        root, tree = build_tree_structure(nodes)
        if root is None:
            print(f"Warning: No valid tree structure found for Result {result_num_actual}")
            continue
        
        # Filter tree by depth
        if max_depth is not None:
            # Debug: show depth distribution before filtering
            depth_counts = {}
            for nid, ndata in tree.items():
                d = ndata['node']['depth']
                depth_counts[d] = depth_counts.get(d, 0) + 1
            print(f"Before filtering - Depth distribution: {dict(sorted(depth_counts.items()))}")
            print(f"Root node depth: {root['depth']}")
            
            tree = filter_tree_by_depth(tree, root, max_depth)
            print(f"Filtering tree to max depth: {max_depth} (from {len(nodes)} nodes to {len(tree)} nodes)")
            
            # Debug: show depth distribution after filtering
            if tree:
                depth_counts_after = {}
                for nid, ndata in tree.items():
                    d = ndata['node']['depth']
                    depth_counts_after[d] = depth_counts_after.get(d, 0) + 1
                print(f"After filtering - Depth distribution: {dict(sorted(depth_counts_after.items()))}")
            else:
                print("WARNING: No nodes after filtering!")
        
        # Determine save path
        if save_path:
            if result_num == 'all':
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                save_path_actual = f"{save_path}_{base_name}_result_{result_num_actual}.{output_format}"
            else:
                # If user-specified save_path has no extension, add format
                if not save_path.endswith(('.png', '.pdf')):
                    save_path_actual = f"{save_path}.{output_format}"
                else:
                    save_path_actual = save_path
        else:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            save_path_actual = f"tree_{base_name}_result_{result_num_actual}.{output_format}"
        
        # Draw tree
        try:
            draw_tree(root, tree, result_num_actual, save_path_actual, max_depth=max_depth, output_format=output_format)
            success_count += 1
        except Exception as e:
            import traceback
            print(f"Error drawing tree for Result {result_num_actual}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description='Visualize tree structure from parse results')
    parser.add_argument('input', help='Path to the main file, file list, or directory')
    parser.add_argument('result_num', nargs='?', default='all', 
                       help='Specific result number to visualize (default: all)')
    parser.add_argument('--save', '-s', help='Save path prefix for the image files')
    parser.add_argument('--list', '-l', action='store_true', 
                       help='Input is a file containing a list of file paths')
    parser.add_argument('--dir', '-d', action='store_true',
                       help='Input is a directory to process all .txt files')
    parser.add_argument('--simplified', action='store_true',
                       help='Force simplified visualization for large trees')
    parser.add_argument('--max-nodes', type=int, default=1000,
                       help='Maximum nodes before switching to simplified view (default: 1000)')
    parser.add_argument('--max-depth', type=int, default=None,
                       help='Maximum depth (number of levels) to visualize (default: all levels)')
    parser.add_argument('--format', type=str, default='pdf', choices=['png', 'pdf'],
                       help='Output image format (default: png, options: png, pdf)')
    
    args = parser.parse_args()
    
    file_paths = []
    
    # Determine input type and get file list
    if args.list:
        # Read from file list
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                file_paths = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: List file does not exist: {args.input}")
            return
    elif args.dir:
        # Process all txt files in directory
        if not os.path.isdir(args.input):
            print(f"Error: Directory does not exist: {args.input}")
            return
        file_paths = glob.glob(os.path.join(args.input, '*.txt'))
        # Filter out time snapshot files
        file_paths = [f for f in file_paths if not re.search(r'_t\d+s\.txt$', f)]
    else:
        # Single file
        file_paths = [args.input]
    
    if not file_paths:
        print("No files to process")
        return
    
    print(f"Processing {len(file_paths)} file(s)...")
    
    # Process each file
    success_count = 0
    for i, filepath in enumerate(file_paths, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(file_paths)}: {os.path.basename(filepath)}")
        print(f"{'='*60}")
        
        if process_single_file(filepath, args.result_num, args.save, args.max_depth, args.format):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{len(file_paths)} files processed successfully")
    print(f"{'='*60}")

def visualize_tree_direct(file_path, result_num=1, save_path=None, max_depth=None, output_format='pdf'):
    """
    Directly visualize tree structure without command line arguments

    Args:
        file_path: File path
        result_num: Result number (default: 1)
        save_path: Save path (optional)
        max_depth: Maximum depth (optional)
        output_format: Output format ('png' or 'pdf', default: 'pdf')
    """
    print(f"Processing file: {file_path}")
    print(f"Result number: {result_num}")

    if save_path:
        print(f"Save path: {save_path}")
    if max_depth:
        print(f"Max depth: {max_depth}")

    success = process_single_file(file_path, result_num, save_path, max_depth, output_format)

    if success:
        print("Visualization complete!")
    else:
        print("Visualization failed!")
    
    return success

def quick_visualize(file_name, prompt_num=1, max_depth=None):
    """
    Quick visualization function - simplest way to use

    Args:
        file_name: File name (can be relative or absolute path)
        prompt_num: Prompt number (default: 1)
        max_depth: Maximum depth limit (optional, None means no limit)
    """
    # If only a file name, try to find it in the result directory
    if not os.path.isabs(file_name) and not os.path.exists(file_name):
        # Search for matching file in result directory
        result_dir = "/home/shuyilin/Jailbreak_oracle/result"
        if os.path.exists(result_dir):
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    if file_name in file and file.endswith('.txt'):
                        file_name = os.path.join(root, file)
                        print(f"Found file: {file_name}")
                        break
    
    # Generate output file name
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    output_name = f"tree_{base_name}_result_{prompt_num}.pdf"
    
    print(f"Starting visualization...")
    print(f"File: {file_name}")
    print(f"Prompt: {prompt_num}")
    if max_depth:
        print(f"Max depth: {max_depth}")
    print(f"Output: {output_name}")
    print("-" * 50)
    
    success = visualize_tree_direct(
        file_path=file_name,
        result_num=prompt_num,
        save_path=output_name,
        max_depth=max_depth,  # Use the passed-in depth limit
        output_format='pdf'
    )
    
    return success

if __name__ == "__main__":
    # Quick usage examples - uncomment and modify the parameters below to use

    # Example 1: Use full file path
    # quick_visualize("/home/shuyilin/Jailbreak_oracle/result/your_file.txt", 1)
    
    # Example 2: Provide only filename, auto-search in result directory
    # quick_visualize("20251018_033631_Llama-3.1-8B-Instruct", 1)
    
    # Example 3: Visualize 3rd prompt
    # quick_visualize("your_file.txt", 3)
    
    # If you want to use command line arguments, comment out the calls above and uncomment the line below:
    main()
