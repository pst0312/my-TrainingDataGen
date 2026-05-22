import unittest

import numpy as np

from spectrum_generator import (
    aes_derivative_profile,
    bremsstrahlung_background,
    edge_profile,
    generate_synthetic_data,
    shirley_background,
    zlp_profile,
)
from SpectDict import ESI_CONFIG


class TestSpectrumGenerator(unittest.TestCase):
    def test_xps_shirley_axis_reversed(self):
        x = np.linspace(0.0, 1400.0, 512)
        y = np.exp(-((x - 400.0) ** 2) / (2 * (8.0 ** 2)))
        x_rev = x[::-1]
        y_rev = y[::-1]
        bg = shirley_background(x_rev, y_rev, n_iter=12, axis_reversed=True)
        self.assertGreater(bg[0], bg[-1], "Shirley background should step up on left side for reversed XPS axis")

    def test_aes_derivative_shape(self):
        x = np.linspace(20.0, 2500.0, 1024)
        y = aes_derivative_profile(x, center=115.0, fwhm=2.0, intensity=1.0, asymmetry=0.3)
        sign_changes = np.sum(np.diff(np.sign(y)) != 0)
        self.assertGreater(sign_changes, 0, "AES derivative should change sign across the dip-spike feature")
        self.assertGreater(np.max(np.abs(y)), 0.2, "AES derivative signal should be visibly non-zero")

    def test_eds_bremsstrahlung_shape(self):
        x = np.linspace(0.1, 20.0, 512)
        bg = bremsstrahlung_background(x, x_min=0.1, intensity=1.0, k=1.7, E0=15.0)
        peak_idx = int(np.argmax(bg))
        peak_energy = x[peak_idx]
        self.assertGreater(peak_energy, 0.5)
        self.assertLess(peak_energy, 5.0)
        self.assertLess(bg[0], 0.35 * bg[peak_idx])
        self.assertLess(bg[-1], 0.2 * bg[peak_idx])

    def test_eels_zlp_and_edge(self):
        x = np.linspace(-5.0, 200.0, 2048)
        zlp = zlp_profile(x, center=0.0, amplitude=5.0, fwhm=0.3)
        center_index = np.argmin(np.abs(x - 0.0))
        self.assertGreater(zlp[center_index], 10 * np.mean(zlp), "ZLP should be intense and narrow at 0 eV")

        edge = edge_profile(x, threshold=80.0, amplitude=1.0, decay_scale=10.0)
        self.assertAlmostEqual(edge[np.argmin(np.abs(x - 0.0))], 0.0, delta=1e-6)
        self.assertGreater(np.max(edge), 0.1)
        self.assertTrue(np.all(edge[x < 80.0] <= 1e-6), "Edge should be zero before the threshold")

    def test_particle_techniques_do_not_generate_trailing_lines(self):
        config = ESI_CONFIG["XPS"]
        spectra = generate_synthetic_data(
            "XPS",
            config,
            material="Gold",
            n_points=512,
            n_lines=1,
            data_complexity=10,
            seed=123,
        )
        self.assertFalse(any(isinstance(k, str) and k.endswith("_trailing") for k in spectra),
                         "Particle techniques should not generate trailing cross-polarized lines")


if __name__ == "__main__":
    unittest.main()
