"""
agent_tools.py
==============
Agent-ready interface for the synthetic spectroscopy data generator.

Provides a clean Python API for AI agents to interact with the spectroscopy
pipeline without CLI overhead. Ensures absolute determinism via seed control
and comprehensive metadata tracking.

Key Classes:
  - SpectroscopyEnvironment: Main interface for generating and evaluating samples
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
from datetime import datetime

from esi_config import ESI_CONFIG, PLOT_STYLE_CONFIG
from material_library import PEAK_LIBRARY, get_material_list, get_techniques_for_material
import spectrum_generator as specgen


class SpectroscopyEnvironment:
    """
    Agent-ready interface for synthetic spectroscopy data generation.
    
    This class provides a deterministic, seed-controlled API for AI agents
    to generate custom spectroscopy samples with precise control over all
    parameters and noise sources.
    
    Attributes:
        output_dir (Path): Directory for generated files
        metadata_log (List[Dict]): Log of all generated samples
    """
    
    def __init__(self, output_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize the spectroscopy environment.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save generated files. If None, creates batch folder
            with timestamp.
        verbose : bool
            If True, print generation details.
        """
        self.verbose = verbose
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"batch_agent_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_log: List[Dict[str, Any]] = []
        
        if self.verbose:
            print(f"✓ SpectroscopyEnvironment initialized")
            print(f"  Output directory: {self.output_dir}/")
    
    def generate_custom_sample(
        self,
        technique: str,
        material: Optional[str] = None,
        vis_complexity: int = 5,
        data_complexity: int = 5,
        seed: Optional[int] = None,
        n_points: int = 2048,
    ) -> Dict[str, Any]:
        """
        Generate a custom spectroscopy sample with deterministic control.
        
        This is the core method for agent interaction. It bypasses CLI entirely
        and returns data directly to the agent.
        
        Parameters
        ----------
        technique : str
            Spectroscopy technique (XPS, AES, EDS, EELS, IR, Raman)
        material : str, optional
            Material name from library. If None, selects randomly.
        vis_complexity : int
            Visual complexity (1-10). Controls image degradation severity.
        data_complexity : int
            Data complexity (1-10). Controls number of peaks and trailing lines.
        seed : int, optional
            Random seed for determinism. If None, uses current time.
            Same seed guarantees identical sample regeneration.
        n_points : int
            Number of data points per spectrum (default 2048).
        
        Returns
        -------
        dict
            Ground-truth metadata with keys:
            - 'sample_id': Unique identifier
            - 'technique': Spectroscopy technique used
            - 'material': Material name
            - 'vis_complexity': Visual complexity score
            - 'data_complexity': Data complexity score
            - 'seed': Random seed used
            - 'trailing_lines': List of trailing line IDs
            - 'csv_path': Path to CSV file
            - 'png_path': Path to PNG image
            - 'data': Pandas DataFrame with spectrum data
            - 'num_lines': Number of primary lines
            - 'base_data_complexity': Technique's base complexity
        
        Raises
        ------
        ValueError
            If technique or material is invalid.
        
        Examples
        --------
        >>> env = SpectroscopyEnvironment()
        >>> result = env.generate_custom_sample(
        ...     technique='XPS',
        ...     material='Silicon (Si)',
        ...     vis_complexity=3,
        ...     data_complexity=5,
        ...     seed=42
        ... )
        >>> print(f"Generated {result['sample_id']}")
        >>> df = result['data']
        >>> img_path = result['png_path']
        """
        # Validate inputs
        if technique not in ESI_CONFIG:
            raise ValueError(f"Unknown technique '{technique}'. Available: {list(ESI_CONFIG.keys())}")
        
        if material is None:
            # Select material randomly from available for this technique
            available = [m for m in PEAK_LIBRARY if technique in PEAK_LIBRARY[m]]
            material = random.choice(available) if available else "Unknown"
        else:
            if material not in PEAK_LIBRARY and material != "Unknown":
                raise ValueError(f"Unknown material '{material}'. Available: {get_material_list()}")
        
        # Clamp complexities
        vis_complexity = max(1, min(vis_complexity, 10))
        data_complexity = max(1, min(data_complexity, 10))
        
        # Set random seeds for absolute determinism
        if seed is None:
            seed = int(datetime.now().timestamp() * 1000) % (2**31)
        
        np.random.seed(seed)
        random.seed(seed)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Generating custom sample:")
            print(f"  Technique: {technique}")
            print(f"  Material: {material}")
            print(f"  Data Complexity: {data_complexity}/10")
            print(f"  Visual Complexity: {vis_complexity}/10")
            print(f"  Seed: {seed}")
        
        # Get configuration
        config = ESI_CONFIG[technique]
        style = PLOT_STYLE_CONFIG[technique]
        base_data_complexity = config.get("base_data_complexity", 5)
        
        # Determine number of lines based on data complexity
        if data_complexity <= 3:
            num_lines = 1
        elif data_complexity <= 5:
            num_lines = np.random.randint(1, 3)
        elif data_complexity <= 7:
            num_lines = np.random.randint(2, 4)
        else:
            num_lines = np.random.randint(3, 6)
        
        # Generate spectral data
        spectra = specgen.generate_synthetic_data(
            technique=technique,
            config=config,
            material=material,
            n_points=n_points,
            n_lines=num_lines,
            data_complexity=data_complexity,
            seed=seed,
        )
        
        # Identify trailing lines
        trailing_lines = [lid for lid in spectra.keys() if isinstance(lid, str) and '_trailing' in str(lid)]
        
        # Create DataFrame with material column
        material_name = material if material else "Unknown"
        df = specgen.create_dataframe(spectra, technique, material=material_name)
        
        # Generate sample ID
        sample_id = f"{technique.lower()}_{material_name.lower().replace(' ', '_')}_{seed}"
        
        # Save CSV
        csv_path = self.output_dir / f"{sample_id}.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate and save plot with visual degradation
        fig, ax = specgen.plot_spectrum(df, spectra, technique, config, style)
        png_path = self.output_dir / f"{sample_id}.png"
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        
        # Apply visual degradation scaled by visual_complexity
        if style.get("low_res", {}).get("enabled", False):
            specgen.apply_visual_degradation(
                str(png_path),
                style.get("low_res", {}),
                visual_complexity=vis_complexity,
                dpi=100
            )
        
        plt.close(fig)
        
        # Create metadata entry
        metadata_entry = {
            "sample_id": sample_id,
            "technique": technique,
            "material": material_name,
            "vis_complexity": vis_complexity,
            "data_complexity": data_complexity,
            "base_data_complexity": base_data_complexity,
            "num_lines": int(num_lines),
            "trailing_lines": trailing_lines,
            "num_trailing_lines": len(trailing_lines),
            "seed": int(seed),
            "csv_path": str(csv_path),
            "png_path": str(png_path),
            "n_points": n_points,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.metadata_log.append(metadata_entry)
        
        if self.verbose:
            print(f"✓ Sample generated: {sample_id}")
            print(f"  Lines: {num_lines} primary + {len(trailing_lines)} trailing")
            print(f"  CSV: {csv_path.name}")
            print(f"  PNG: {png_path.name}")
        
        # Return complete ground-truth dict for agent
        return {
            "sample_id": sample_id,
            "technique": technique,
            "material": material_name,
            "vis_complexity": vis_complexity,
            "data_complexity": data_complexity,
            "base_data_complexity": base_data_complexity,
            "seed": int(seed),
            "trailing_lines": trailing_lines,
            "num_lines": int(num_lines),
            "num_trailing_lines": len(trailing_lines),
            "csv_path": str(csv_path),
            "png_path": str(png_path),
            "data": df,
            "spectra": spectra,
        }
    
    def batch_generate(
        self,
        samples: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple samples with specified parameters.
        
        Parameters
        ----------
        samples : List[Dict]
            List of sample specifications. Each dict should have keys:
            - 'technique' (str, required)
            - 'material' (str, optional)
            - 'vis_complexity' (int, optional, default 5)
            - 'data_complexity' (int, optional, default 5)
            - 'seed' (int, optional)
        verbose : bool
            Print progress for each sample.
        
        Returns
        -------
        List[Dict]
            List of metadata dicts for each generated sample.
        
        Example
        -------
        >>> env = SpectroscopyEnvironment()
        >>> samples = [
        ...     {'technique': 'XPS', 'material': 'Silicon (Si)', 'data_complexity': 5, 'seed': 1},
        ...     {'technique': 'XPS', 'material': 'Gold (Au)', 'data_complexity': 5, 'seed': 2},
        ...     {'technique': 'EDS', 'vis_complexity': 8, 'data_complexity': 2, 'seed': 3},
        ... ]
        >>> results = env.batch_generate(samples)
        """
        results = []
        for i, spec in enumerate(samples, 1):
            if verbose:
                print(f"\n[{i}/{len(samples)}]", end=" ")
            
            result = self.generate_custom_sample(
                technique=spec.get('technique'),
                material=spec.get('material'),
                vis_complexity=spec.get('vis_complexity', 5),
                data_complexity=spec.get('data_complexity', 5),
                seed=spec.get('seed'),
                n_points=spec.get('n_points', 2048),
            )
            results.append(result)
        
        return results
    
    def save_metadata(self) -> str:
        """
        Save comprehensive metadata manifest to JSON.
        
        Returns
        -------
        str
            Path to saved metadata.json file.
        """
        metadata_path = self.output_dir / "metadata.json"
        
        metadata = {
            "batch_info": {
                "output_dir": str(self.output_dir),
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(self.metadata_log),
            },
            "samples": self.metadata_log,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"\n✓ Metadata saved: {metadata_path}")
        
        return str(metadata_path)
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from existing batch.
        
        Returns
        -------
        dict
            Loaded metadata manifest.
        """
        metadata_path = self.output_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_sample(self, sample_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a specific sample by ID.
        
        Parameters
        ----------
        sample_id : str
            Sample identifier.
        
        Returns
        -------
        dict
            Sample metadata entry.
        """
        for entry in self.metadata_log:
            if entry['sample_id'] == sample_id:
                return entry
        
        raise ValueError(f"Sample '{sample_id}' not found in metadata log")
    
    def list_samples(self) -> List[str]:
        """
        List all sample IDs in current batch.
        
        Returns
        -------
        List[str]
            Sample identifiers.
        """
        return [entry['sample_id'] for entry in self.metadata_log]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over all generated samples.
        
        Returns
        -------
        dict
            Statistics including complexity distributions, material counts, etc.
        """
        if not self.metadata_log:
            return {"num_samples": 0}
        
        complexities_data = [entry['data_complexity'] for entry in self.metadata_log]
        complexities_vis = [entry['vis_complexity'] for entry in self.metadata_log]
        techniques = [entry['technique'] for entry in self.metadata_log]
        materials = [entry['material'] for entry in self.metadata_log]
        trailing_counts = [entry['num_trailing_lines'] for entry in self.metadata_log]
        
        return {
            "num_samples": len(self.metadata_log),
            "data_complexity": {
                "mean": float(np.mean(complexities_data)),
                "std": float(np.std(complexities_data)),
                "min": int(np.min(complexities_data)),
                "max": int(np.max(complexities_data)),
            },
            "visual_complexity": {
                "mean": float(np.mean(complexities_vis)),
                "std": float(np.std(complexities_vis)),
                "min": int(np.min(complexities_vis)),
                "max": int(np.max(complexities_vis)),
            },
            "techniques": dict(zip(*np.unique(techniques, return_counts=True))),
            "materials": dict(zip(*np.unique(materials, return_counts=True))),
            "trailing_lines": {
                "mean": float(np.mean(trailing_counts)),
                "total": int(np.sum(trailing_counts)),
                "max": int(np.max(trailing_counts)),
            },
        }


def create_environment(output_dir: Optional[str] = None) -> SpectroscopyEnvironment:
    """
    Convenience function to create a spectroscopy environment.
    
    Parameters
    ----------
    output_dir : str, optional
        Output directory for generated files.
    
    Returns
    -------
    SpectroscopyEnvironment
        Initialized environment ready for sample generation.
    """
    return SpectroscopyEnvironment(output_dir=output_dir, verbose=True)
