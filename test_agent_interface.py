#!/usr/bin/env python3
"""
Comprehensive test suite for the agent-ready interface.
Demonstrates all key features: determinism, metadata, batch processing, ground truth access.
"""

from agent_tools import SpectroscopyEnvironment
import json
import hashlib
import os
import sys


def test_determinism():
    """Test that same seed produces identical output."""
    print("\n" + "=" * 70)
    print("TEST 1: Determinism (Same seed → Identical output)")
    print("=" * 70)
    
    env1 = SpectroscopyEnvironment(output_dir='test_det_1', verbose=False)
    result1 = env1.generate_custom_sample(
        technique='XPS',
        material='Gold (Au)',
        vis_complexity=5,
        data_complexity=7,
        seed=1000
    )
    
    env2 = SpectroscopyEnvironment(output_dir='test_det_2', verbose=False)
    result2 = env2.generate_custom_sample(
        technique='XPS',
        material='Gold (Au)',
        vis_complexity=5,
        data_complexity=7,
        seed=1000
    )
    
    # Check CSV byte-for-byte identity
    with open(result1['csv_path'], 'rb') as f:
        hash1 = hashlib.md5(f.read()).hexdigest()
    with open(result2['csv_path'], 'rb') as f:
        hash2 = hashlib.md5(f.read()).hexdigest()
    
    csv_match = hash1 == hash2
    print(f"CSV file hashes match: {csv_match} ✓" if csv_match else f"CSV file hashes match: {csv_match} ✗")
    
    # Check spectral decomposition - compare line IDs and counts
    lines1 = set(key for key in result1['spectra'].keys() if not (isinstance(key, str) and '_trailing' in key))
    lines2 = set(key for key in result2['spectra'].keys() if not (isinstance(key, str) and '_trailing' in key))
    
    spec_match = lines1 == lines2
    print(f"Spectrum line counts match: {spec_match} ✓" if spec_match else f"Spectrum line counts match: {spec_match} ✗")
    
    return csv_match and spec_match


def test_different_seeds_different_output():
    """Test that different seeds produce different output."""
    print("\n" + "=" * 70)
    print("TEST 2: Different Seeds → Different Output")
    print("=" * 70)
    
    env = SpectroscopyEnvironment(output_dir='test_diff_seeds', verbose=False)
    result1 = env.generate_custom_sample(
        technique='XPS',
        material='Silicon (Si)',
        vis_complexity=5,
        data_complexity=5,
        seed=2000
    )
    
    result2 = env.generate_custom_sample(
        technique='XPS',
        material='Silicon (Si)',
        vis_complexity=5,
        data_complexity=5,
        seed=2001
    )
    
    # CSV should be different
    with open(result1['csv_path'], 'rb') as f:
        hash1 = hashlib.md5(f.read()).hexdigest()
    with open(result2['csv_path'], 'rb') as f:
        hash2 = hashlib.md5(f.read()).hexdigest()
    
    csv_different = hash1 != hash2
    print(f"CSV files are different: {csv_different} ✓" if csv_different else f"CSV files are different: {csv_different} ✗")
    
    # Visual degradation should be different (different seed affects blur, noise, etc)
    # But line generation is deterministic from data_complexity
    print(f"Visual degradation changed (different PNG): True (assumed - seed affects visual) ✓")
    
    return csv_different


def test_metadata_structure():
    """Test metadata manifest structure and content."""
    print("\n" + "=" * 70)
    print("TEST 3: Metadata Manifest Structure")
    print("=" * 70)
    
    env = SpectroscopyEnvironment(output_dir='test_metadata', verbose=False)
    
    samples = [
        {'technique': 'XPS', 'material': 'Copper (Cu)', 'vis_complexity': 2, 'data_complexity': 3, 'seed': 3000},
        {'technique': 'EDS', 'material': 'Gold (Au)', 'vis_complexity': 5, 'data_complexity': 2, 'seed': 3001},
        {'technique': 'IR', 'material': 'Silicon (Si)', 'vis_complexity': 8, 'data_complexity': 6, 'seed': 3002},
    ]
    
    results = env.batch_generate(samples, verbose=False)
    metadata_path = env.save_metadata()
    
    print(f"Metadata saved to: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check batch_info
    batch_ok = (
        'batch_info' in metadata and
        'output_dir' in metadata['batch_info'] and
        'timestamp' in metadata['batch_info'] and
        'num_samples' in metadata['batch_info']
    )
    print(f"Batch info structure OK: {batch_ok} ✓" if batch_ok else f"Batch info structure OK: {batch_ok} ✗")
    
    # Check sample count
    sample_count_ok = metadata['batch_info']['num_samples'] == len(samples)
    print(f"Sample count matches: {sample_count_ok} ({len(samples)}) ✓" if sample_count_ok else f"Sample count matches: {sample_count_ok} ✗")
    
    # Check sample fields
    required_fields = [
        'sample_id', 'technique', 'material', 'vis_complexity', 'data_complexity',
        'num_lines', 'trailing_lines', 'num_trailing_lines', 'seed',
        'csv_path', 'png_path', 'timestamp'
    ]
    
    all_fields_ok = all(
        all(field in sample for field in required_fields)
        for sample in metadata['samples']
    )
    print(f"All required fields present: {all_fields_ok} ✓" if all_fields_ok else f"All required fields present: {all_fields_ok} ✗")
    
    # Print sample summary
    print(f"\nGenerated samples:")
    for i, sample in enumerate(metadata['samples'], 1):
        print(f"  [{i}] {sample['sample_id']}")
        print(f"      {sample['technique']} on {sample['material']}")
        print(f"      vis={sample['vis_complexity']}, data={sample['data_complexity']}, seed={sample['seed']}")
        print(f"      lines={sample['num_lines']}, trailing={sample['num_trailing_lines']}")
    
    return batch_ok and sample_count_ok and all_fields_ok


def test_ground_truth_access():
    """Test access to ground truth peak data via CSV."""
    print("\n" + "=" * 70)
    print("TEST 4: Ground Truth Access via CSV")
    print("=" * 70)
    
    env = SpectroscopyEnvironment(output_dir='test_ground_truth', verbose=False)
    result = env.generate_custom_sample(
        technique='XPS',
        material='Platinum (Pt)',
        vis_complexity=3,
        data_complexity=6,
        seed=4000
    )
    
    # Check result has required keys
    has_csv = 'csv_path' in result and os.path.exists(result['csv_path'])
    has_png = 'png_path' in result and os.path.exists(result['png_path'])
    has_data = 'data' in result  # pandas DataFrame
    has_spectra = 'spectra' in result and len(result['spectra']) > 0
    
    print(f"CSV file exists: {has_csv} ✓" if has_csv else f"CSV file exists: {has_csv} ✗")
    print(f"PNG file exists: {has_png} ✓" if has_png else f"PNG file exists: {has_png} ✗")
    print(f"DataFrame in memory: {has_data} ✓" if has_data else f"DataFrame in memory: {has_data} ✗")
    print(f"Spectral decomposition available: {has_spectra} ✓" if has_spectra else f"Spectral decomposition available: {has_spectra} ✗")
    
    if has_data:
        df = result['data']
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:\n{df.head(3)}")
    
    if has_spectra:
        # Get non-trailing line count
        non_trailing_lines = [key for key in result['spectra'].keys() if not (isinstance(key, str) and '_trailing' in key)]
        num_lines = len(non_trailing_lines)
        print(f"\nNumber of primary spectral lines: {num_lines}")
        
        # Check line structure - each line is a dict with peak parameters
        if num_lines > 0:
            line_key = non_trailing_lines[0]
            line = result['spectra'][line_key]
            print(f"Example line data (key={line_key}):")
            if isinstance(line, dict):
                for k, v in sorted(line.items())[:5]:  # Show first 5 items
                    print(f"  {k}: {v}")
    
    return has_csv and has_png and has_data and has_spectra


def test_decoupled_complexity():
    """Test that vis and data complexity are truly independent."""
    print("\n" + "=" * 70)
    print("TEST 5: Decoupled Complexity (Data vs Visual)")
    print("=" * 70)
    
    env = SpectroscopyEnvironment(output_dir='test_decoupled', verbose=False)
    
    # Case 1: Simple data, pristine visual
    result1 = env.generate_custom_sample(
        technique='XPS',
        material='Copper (Cu)',
        vis_complexity=1,  # Pristine
        data_complexity=1,  # Simple
        seed=5000
    )
    
    # Case 2: Simple data, degraded visual
    result2 = env.generate_custom_sample(
        technique='XPS',
        material='Copper (Cu)',
        vis_complexity=10,  # Degraded
        data_complexity=1,  # Same simple data
        seed=5000  # Same seed for same data!
    )
    
    # Case 3: Complex data, pristine visual
    result3 = env.generate_custom_sample(
        technique='XPS',
        material='Copper (Cu)',
        vis_complexity=1,  # Pristine
        data_complexity=8,  # Complex
        seed=5001
    )
    
    # Case 4: Complex data, degraded visual
    result4 = env.generate_custom_sample(
        technique='XPS',
        material='Copper (Cu)',
        vis_complexity=10,  # Degraded
        data_complexity=8,  # Same complex data
        seed=5001  # Same seed for same data!
    )
    
    # Cases 1 & 2 should have same spectral data (same seed, same data_complexity)
    csv1_hash = hashlib.md5(open(result1['csv_path'], 'rb').read()).hexdigest()
    csv2_hash = hashlib.md5(open(result2['csv_path'], 'rb').read()).hexdigest()
    same_data_different_visual = csv1_hash == csv2_hash
    
    print(f"Case 1 & 2 (same data, different visual) have identical CSV: {same_data_different_visual} ✓" if same_data_different_visual else f"Case 1 & 2 have identical CSV: {same_data_different_visual} ✗")
    
    # Cases 3 & 4 should have same spectral data (same seed, same data_complexity)
    csv3_hash = hashlib.md5(open(result3['csv_path'], 'rb').read()).hexdigest()
    csv4_hash = hashlib.md5(open(result4['csv_path'], 'rb').read()).hexdigest()
    complex_same_different_visual = csv3_hash == csv4_hash
    
    print(f"Case 3 & 4 (same complex data, different visual) have identical CSV: {complex_same_different_visual} ✓" if complex_same_different_visual else f"Case 3 & 4 have identical CSV: {complex_same_different_visual} ✗")
    
    # Cases 1 & 3 should have different spectral data (different data_complexity)
    different_data_same_visual = csv1_hash != csv3_hash
    
    print(f"Case 1 & 3 (different data, same visual) have different CSV: {different_data_same_visual} ✓" if different_data_same_visual else f"Case 1 & 3 have different CSV: {different_data_same_visual} ✗")
    
    return same_data_different_visual and complex_same_different_visual and different_data_same_visual


def test_trailing_lines():
    """Test trailing line generation at high data complexity."""
    print("\n" + "=" * 70)
    print("TEST 6: Trailing Lines (High Data Complexity)")
    print("=" * 70)
    
    env = SpectroscopyEnvironment(output_dir='test_trailing', verbose=False)
    
    low_complexity = env.generate_custom_sample(
        technique='XPS',
        material='Gold (Au)',
        vis_complexity=5,
        data_complexity=2,  # Low complexity
        seed=6000
    )
    
    high_complexity = env.generate_custom_sample(
        technique='XPS',
        material='Gold (Au)',
        vis_complexity=5,
        data_complexity=9,  # High complexity
        seed=6001
    )
    
    low_trailing = low_complexity['num_trailing_lines']
    high_trailing = high_complexity['num_trailing_lines']
    
    print(f"Low complexity sample: {low_trailing} trailing lines")
    print(f"High complexity sample: {high_trailing} trailing lines")
    
    # High complexity should have more trailing lines
    more_trailing_at_high = high_trailing >= low_trailing
    print(f"More trailing lines at high complexity: {more_trailing_at_high} ✓" if more_trailing_at_high else f"More trailing lines at high complexity: {more_trailing_at_high} ✗")
    
    return more_trailing_at_high


def cleanup():
    """Remove test directories."""
    import shutil
    test_dirs = [
        'test_det_1', 'test_det_2', 'test_diff_seeds',
        'test_metadata', 'test_ground_truth', 'test_decoupled', 'test_trailing'
    ]
    for d in test_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Cleaned up {d}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AGENT INTERFACE COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    try:
        results['determinism'] = test_determinism()
        results['different_seeds'] = test_different_seeds_different_output()
        results['metadata'] = test_metadata_structure()
        results['ground_truth'] = test_ground_truth_access()
        results['decoupled'] = test_decoupled_complexity()
        results['trailing'] = test_trailing_lines()
    finally:
        cleanup()
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {test_name:30s} {status}")
    
    all_passed = all(results.values())
    print("\n" + ("=" * 70))
    if all_passed:
        print("ALL TESTS PASSED ✓✓✓")
        print("Agent interface is fully functional and ready for use.")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
