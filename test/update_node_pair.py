#!/usr/bin/env python3
"""
Update all test files to use the new recommended node pair.
"""

import os
import re

def update_node_pair_in_file(file_path, old_source, old_dest, new_source, new_dest):
    """Update node pair in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace the old node pair with the new one
        updated_content = content.replace(old_source, new_source)
        updated_content = updated_content.replace(old_dest, new_dest)
        
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all test files with the new node pair."""
    
    # Old and new node pairs
    old_source = "10.2.1.105"
    old_dest = "10.2.1.4"
    new_source = "10.1.1.109"
    new_dest = "10.1.1.5"
    
    print("Updating Node Pair in Test Files")
    print("=" * 50)
    print(f"Old pair: {old_source} → {old_dest}")
    print(f"New pair: {new_source} → {new_dest}")
    print()
    
    # Files to update
    test_files = [
        "test/test_ucsb_advanced_10000.py",
        "test/test_ucsb_10000_rounds.py",
        "test/test_ucsb_10000_rounds_extended.py",
        "test/test_ucsb_long_run.py",
        "test/test_ucsb_regret_verification.py",
        "test/test_ucsb_original.py",
        "test/test_ucsb_ctsb_simple.py",
        "test/check_ucsb_regret.py",
        "test/plot_ucsb_regret_comparison.py",
        "example/example_ucsb_meshnet_memmap.py"
    ]
    
    updated_count = 0
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"Updating {file_path}...")
            if update_node_pair_in_file(file_path, old_source, old_dest, new_source, new_dest):
                print(f"  ✓ Updated successfully")
                updated_count += 1
            else:
                print(f"  ✗ Failed to update")
        else:
            print(f"Skipping {file_path} (file not found)")
    
    print(f"\nUpdate completed: {updated_count}/{len(test_files)} files updated")
    
    # Also update the environment file if needed
    env_file = "src/environments/ucsb_meshnet_memmap.py"
    if os.path.exists(env_file):
        print(f"\nChecking {env_file} for default node selection...")
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check if the file has hardcoded default selection
        if "self.nodes[0], self.nodes[-1]" in content:
            print("  Note: Environment uses dynamic node selection (first and last alphabetically)")
            print("  This will automatically use the new nodes if they are first/last in sorted order")
        else:
            print("  No hardcoded default selection found")

if __name__ == "__main__":
    main() 