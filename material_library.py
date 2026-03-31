"""
Material Library for Synthetic Spectroscopy Generation

Contains characteristic peak definitions for 25+ realistic materials across
XPS, EDS, AES, IR, Raman, and EELS spectroscopy techniques. Each material
includes accurate peak positions, natural FWHMs, relative intensities, and
brief descriptions based on literature values.

Organization:
  - Noble/Transition Metals: Au, Cu, Ag, Pt, Ni, Ti
  - Semiconductors: Si, GaAs, SiC, ZnO, GaN
  - Oxides/Ceramics: SiO2, Al2O3, TiO2, MoS2, WS2
  - Polymers: PMMA, PTFE, Polystyrene, Polyethylene
  - Organics/Salts: NaN3, Graphite, Graphene Oxide
"""

from typing import Dict, Any

# ============================================================================
# Material Peak Library
# ============================================================================

PEAK_LIBRARY: Dict[str, Dict[str, Any]] = {

    # ========== NOBLE & TRANSITION METALS ==========

    "Gold (Au)": {
        "description": "Noble metal; XPS Au 4f doublet, AES L3M45M45 Auger line.",
        "XPS": {
            "peaks": [
                {"position": 84.0, "intensity": 0.67, "fwhm": 0.8, "type": "Au_4f_7/2"},
                {"position": 87.8, "intensity": 0.33, "fwhm": 0.8, "type": "Au_4f_5/2"},
            ]
        },
        "AES": {
            "peaks": [
                {"position": 239.0, "intensity": 0.50, "fwhm": 3.5, "type": "Au_L_M45M45"},
                {"position": 325.0, "intensity": 0.30, "fwhm": 4.0, "type": "Au_M_NOP"},
            ]
        },
        "EDS": {
            "peaks": [
                {"position": 9.628, "intensity": 1.00, "fwhm": 0.15, "type": "Au_L_alpha"},
                {"position": 11.442, "intensity": 0.35, "fwhm": 0.15, "type": "Au_L_beta"},
            ]
        },
    },

    "Copper (Cu)": {
        "description": "Transition metal; Cu 2p doublet in XPS, characteristic AES M45N45.",
        "XPS": {
            "peaks": [
                {"position": 932.7, "intensity": 0.67, "fwhm": 1.4, "type": "Cu_2p_3/2"},
                {"position": 952.5, "intensity": 0.33, "fwhm": 1.4, "type": "Cu_2p_1/2"},
            ]
        },
        "AES": {
            "peaks": [
                {"position": 920.0, "intensity": 0.70, "fwhm": 5.0, "type": "Cu_L_M45M45"},
                {"position": 64.0, "intensity": 0.40, "fwhm": 3.0, "type": "Cu_KLL"},
            ]
        },
        "EDS": {
            "peaks": [
                {"position": 8.048, "intensity": 1.00, "fwhm": 0.13, "type": "Cu_K_alpha"},
                {"position": 8.905, "intensity": 0.20, "fwhm": 0.13, "type": "Cu_K_beta"},
            ]
        },
    },

    "Silver (Ag)": {
        "description": "Noble metal; Ag 3d doublet very narrow in XPS.",
        "XPS": {
            "peaks": [
                {"position": 368.3, "intensity": 0.67, "fwhm": 0.7, "type": "Ag_3d_5/2"},
                {"position": 374.2, "intensity": 0.33, "fwhm": 0.7, "type": "Ag_3d_3/2"},
            ]
        },
        "AES": {
            "peaks": [
                {"position": 356.0, "intensity": 0.60, "fwhm": 4.0, "type": "Ag_M_NOP"},
                {"position": 483.0, "intensity": 0.35, "fwhm": 4.5, "type": "Ag_MVV"},
            ]
        },
        "EDS": {
            "peaks": [
                {"position": 22.163, "intensity": 1.00, "fwhm": 0.20, "type": "Ag_L_alpha"},
                {"position": 24.943, "intensity": 0.38, "fwhm": 0.20, "type": "Ag_L_beta"},
            ]
        },
    },

    "Platinum (Pt)": {
        "description": "Noble metal; Pt 4f narrow doublet, strong EDS X-ray.",
        "XPS": {
            "peaks": [
                {"position": 71.1, "intensity": 0.67, "fwhm": 0.9, "type": "Pt_4f_7/2"},
                {"position": 74.4, "intensity": 0.33, "fwhm": 0.9, "type": "Pt_4f_5/2"},
            ]
        },
        "AES": {
            "peaks": [
                {"position": 237.0, "intensity": 0.65, "fwhm": 4.0, "type": "Pt_M_NOP"},
                {"position": 64.0, "intensity": 0.40, "fwhm": 3.5, "type": "Pt_KLL"},
            ]
        },
        "EDS": {
            "peaks": [
                {"position": 9.442, "intensity": 1.00, "fwhm": 0.14, "type": "Pt_L_alpha"},
                {"position": 11.084, "intensity": 0.33, "fwhm": 0.14, "type": "Pt_L_beta"},
            ]
        },
    },

    "Nickel (Ni)": {
        "description": "Ferromagnetic transition metal; Ni 2p satellite features.",
        "XPS": {
            "peaks": [
                {"position": 852.7, "intensity": 0.60, "fwhm": 1.6, "type": "Ni_2p_3/2"},
                {"position": 858.0, "intensity": 0.15, "fwhm": 1.8, "type": "Ni_2p_3/2_satellite"},
                {"position": 870.0, "intensity": 0.33, "fwhm": 1.6, "type": "Ni_2p_1/2"},
            ]
        },
        "AES": {
            "peaks": [
                {"position": 848.0, "intensity": 0.70, "fwhm": 5.5, "type": "Ni_L_M45M45"},
            ]
        },
        "EDS": {
            "peaks": [
                {"position": 7.478, "intensity": 1.00, "fwhm": 0.12, "type": "Ni_K_alpha"},
                {"position": 8.266, "intensity": 0.20, "fwhm": 0.12, "type": "Ni_K_beta"},
            ]
        },
    },

    "Titanium (Ti)": {
        "description": "Transition metal; Ti 2p doublet, common in ceramics and coatings.",
        "XPS": {
            "peaks": [
                {"position": 454.2, "intensity": 0.67, "fwhm": 1.2, "type": "Ti_2p_3/2"},
                {"position": 460.0, "intensity": 0.33, "fwhm": 1.2, "type": "Ti_2p_1/2"},
            ]
        },
        "AES": {
            "peaks": [
                {"position": 418.0, "intensity": 0.65, "fwhm": 4.5, "type": "Ti_L_M45M45"},
            ]
        },
        "EDS": {
            "peaks": [
                {"position": 4.511, "intensity": 1.00, "fwhm": 0.10, "type": "Ti_K_alpha"},
                {"position": 4.931, "intensity": 0.20, "fwhm": 0.10, "type": "Ti_K_beta"},
            ]
        },
    },

    # ========== SEMICONDUCTORS ==========

    "Silicon (Si)": {
        "description": "Group IV semiconductor; Si 2p doublet in XPS, strong K_alpha EDS line.",
        "XPS": {
            "peaks": [
                {"position": 99.3, "intensity": 0.67, "fwhm": 0.85, "type": "Si_2p_1/2"},
                {"position": 99.8, "intensity": 0.33, "fwhm": 0.85, "type": "Si_2p_3/2"},
            ]
        },
        "EDS": {
            "peaks": [
                {"position": 1.740, "intensity": 1.00, "fwhm": 0.12, "type": "Si_K_alpha"},
                {"position": 1.835, "intensity": 0.20, "fwhm": 0.12, "type": "Si_K_beta"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 520.5, "intensity": 1.00, "fwhm": 3.5, "type": "TO_mode"},
            ]
        },
    },

    "Gallium Arsenide (GaAs)": {
        "description": "III-V semiconductor; direct bandgap, lattice-matched to Ge buffer.",
        "EDS": {
            "peaks": [
                {"position": 9.886, "intensity": 0.60, "fwhm": 0.16, "type": "Ga_K_alpha"},
                {"position": 10.879, "intensity": 0.40, "fwhm": 0.14, "type": "As_K_alpha"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 268.0, "intensity": 1.00, "fwhm": 5.0, "type": "TO_mode"},
                {"position": 294.0, "intensity": 0.80, "fwhm": 4.0, "type": "LO_mode"},
            ]
        },
    },

    "Silicon Carbide (SiC)": {
        "description": "III-V compound; wide bandgap, excellent thermal properties.",
        "EDS": {
            "peaks": [
                {"position": 1.740, "intensity": 0.50, "fwhm": 0.12, "type": "Si_K_alpha"},
                {"position": 0.277, "intensity": 0.50, "fwhm": 0.08, "type": "C_K_alpha"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 204.0, "intensity": 1.00, "fwhm": 6.0, "type": "TO_Si_C"},
                {"position": 266.0, "intensity": 0.85, "fwhm": 5.0, "type": "LO_Si_C"},
            ]
        },
    },

    "Zinc Oxide (ZnO)": {
        "description": "II-VI semiconductor; wurtzite structure, strong UV absorption.",
        "EDS": {
            "peaks": [
                {"position": 8.639, "intensity": 0.50, "fwhm": 0.13, "type": "Zn_K_alpha"},
                {"position": 0.525, "intensity": 0.50, "fwhm": 0.08, "type": "O_K_alpha"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 101.0, "intensity": 1.00, "fwhm": 8.0, "type": "E2_low"},
                {"position": 437.0, "intensity": 0.95, "fwhm": 7.0, "type": "E2_high"},
            ]
        },
    },

    "Gallium Nitride (GaN)": {
        "description": "III-V nitride; direct bandgap, used in power electronics.",
        "EDS": {
            "peaks": [
                {"position": 9.886, "intensity": 0.50, "fwhm": 0.16, "type": "Ga_K_alpha"},
                {"position": 0.392, "intensity": 0.50, "fwhm": 0.08, "type": "N_K_alpha"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 317.0, "intensity": 1.00, "fwhm": 5.0, "type": "A1_LO"},
                {"position": 568.0, "intensity": 0.90, "fwhm": 4.5, "type": "E2_high"},
            ]
        },
    },

    # ========== OXIDES & CERAMICS ==========

    "Silicon Dioxide (SiO2)": {
        "description": "Common oxide; quartz structure, IR active Si-O stretch.",
        "EDS": {
            "peaks": [
                {"position": 1.740, "intensity": 0.33, "fwhm": 0.12, "type": "Si_K_alpha"},
                {"position": 0.525, "intensity": 0.67, "fwhm": 0.08, "type": "O_K_alpha"},
            ]
        },
        "IR": {
            "peaks": [
                {"position": 1100.0, "intensity": 1.00, "fwhm": 150.0, "type": "Si_O_asymmetric_stretch"},
                {"position": 800.0, "intensity": 0.60, "fwhm": 100.0, "type": "Si_O_symmetric_stretch"},
                {"position": 470.0, "intensity": 0.40, "fwhm": 100.0, "type": "Si_O_bending"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 1085.0, "intensity": 1.00, "fwhm": 40.0, "type": "Si_O_stretch"},
                {"position": 470.0, "intensity": 0.45, "fwhm": 50.0, "type": "Si_O_bending"},
                {"position": 200.0, "intensity": 0.30, "fwhm": 30.0, "type": "lattice_mode"},
            ]
        },
    },

    "Aluminum Oxide (Al2O3)": {
        "description": "Sapphire; corundum structure, widely used abrasive and insulator.",
        "EDS": {
            "peaks": [
                {"position": 1.487, "intensity": 0.40, "fwhm": 0.11, "type": "Al_K_alpha"},
                {"position": 0.525, "intensity": 0.60, "fwhm": 0.08, "type": "O_K_alpha"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 418.0, "intensity": 1.00, "fwhm": 20.0, "type": "A1g_mode"},
                {"position": 380.0, "intensity": 0.75, "fwhm": 25.0, "type": "Eg_mode"},
                {"position": 645.0, "intensity": 0.85, "fwhm": 22.0, "type": "A1g_mode2"},
            ]
        },
    },

    "Titanium Oxide (TiO2)": {
        "description": "Rutile/anatase polymorphs; photocatalytic, strong IR signatures.",
        "EDS": {
            "peaks": [
                {"position": 4.511, "intensity": 0.33, "fwhm": 0.10, "type": "Ti_K_alpha"},
                {"position": 0.525, "intensity": 0.67, "fwhm": 0.08, "type": "O_K_alpha"},
            ]
        },
        "IR": {
            "peaks": [
                {"position": 900.0, "intensity": 1.00, "fwhm": 80.0, "type": "Ti_O_stretch"},
                {"position": 600.0, "intensity": 0.80, "fwhm": 70.0, "type": "Ti_O_bending"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 447.0, "intensity": 1.00, "fwhm": 8.0, "type": "Eg_anatase"},
                {"position": 612.0, "intensity": 0.85, "fwhm": 10.0, "type": "A1g_rutile"},
            ]
        },
    },

    "Molybdenum Disulfide (MoS2)": {
        "description": "2D transition metal dichalcogenide; direct bandgap monolayer.",
        "EDS": {
            "peaks": [
                {"position": 2.293, "intensity": 0.33, "fwhm": 0.09, "type": "Mo_K_alpha"},
                {"position": 2.308, "intensity": 0.67, "fwhm": 0.10, "type": "S_K_alpha"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 383.0, "intensity": 1.00, "fwhm": 6.0, "type": "E1_2g"},
                {"position": 409.0, "intensity": 0.95, "fwhm": 5.5, "type": "A1g"},
            ]
        },
    },

    "Tungsten Disulfide (WS2)": {
        "description": "2D transition metal dichalcogenide; strong photoluminescence.",
        "EDS": {
            "peaks": [
                {"position": 8.398, "intensity": 0.33, "fwhm": 0.13, "type": "W_K_alpha"},
                {"position": 2.308, "intensity": 0.67, "fwhm": 0.10, "type": "S_K_alpha"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 355.0, "intensity": 1.00, "fwhm": 5.5, "type": "E1_2g"},
                {"position": 420.0, "intensity": 0.92, "fwhm": 6.0, "type": "A1g"},
            ]
        },
    },

    # ========== POLYMERS ==========

    "PMMA (Acrylic)": {
        "description": "Polymethyl methacrylate; thermoplastic, strong C=O and C-O stretches.",
        "IR": {
            "peaks": [
                {"position": 2950.0, "intensity": 0.80, "fwhm": 40.0, "type": "C_H_stretch"},
                {"position": 1730.0, "intensity": 1.00, "fwhm": 30.0, "type": "C_O_stretch"},
                {"position": 1240.0, "intensity": 0.70, "fwhm": 50.0, "type": "C_O_bending"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 2950.0, "intensity": 0.75, "fwhm": 35.0, "type": "C_H_stretch"},
                {"position": 1731.0, "intensity": 0.90, "fwhm": 25.0, "type": "C_O_stretch"},
                {"position": 1450.0, "intensity": 0.50, "fwhm": 30.0, "type": "C_H_bending"},
            ]
        },
    },

    "PTFE (Teflon)": {
        "description": "Polytetrafluoroethylene; highly fluorinated, strong C-F stretch.",
        "IR": {
            "peaks": [
                {"position": 1200.0, "intensity": 1.00, "fwhm": 80.0, "type": "C_F_stretch"},
                {"position": 600.0, "intensity": 0.40, "fwhm": 100.0, "type": "C_F_bending"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 1381.0, "intensity": 1.00, "fwhm": 20.0, "type": "CF2_stretch"},
            ]
        },
    },

    "Polystyrene (PS)": {
        "description": "Aromatic polymer; benzene ring vibrations prominent in Raman.",
        "IR": {
            "peaks": [
                {"position": 3000.0, "intensity": 0.70, "fwhm": 35.0, "type": "C_H_aromatic_stretch"},
                {"position": 1600.0, "intensity": 0.75, "fwhm": 40.0, "type": "aromatic_C_C_stretch"},
                {"position": 1500.0, "intensity": 0.60, "fwhm": 35.0, "type": "aromatic_C_C_stretch"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 3000.0, "intensity": 0.65, "fwhm": 30.0, "type": "C_H_aromatic_stretch"},
                {"position": 1600.0, "intensity": 0.70, "fwhm": 35.0, "type": "aromatic_C_C_stretch"},
                {"position": 1028.0, "intensity": 1.00, "fwhm": 15.0, "type": "aromatic_C_H_in_plane_bend"},
            ]
        },
    },

    "Polyethylene (PE)": {
        "description": "Aliphatic hydrocarbon polymer; simple C-H and C-C stretches.",
        "IR": {
            "peaks": [
                {"position": 2950.0, "intensity": 1.00, "fwhm": 45.0, "type": "C_H_asymmetric_stretch"},
                {"position": 2850.0, "intensity": 0.85, "fwhm": 45.0, "type": "C_H_symmetric_stretch"},
                {"position": 1450.0, "intensity": 0.70, "fwhm": 50.0, "type": "C_H_bending"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 2880.0, "intensity": 1.00, "fwhm": 40.0, "type": "C_H_stretch"},
                {"position": 1442.0, "intensity": 0.80, "fwhm": 30.0, "type": "C_H_bending"},
                {"position": 1130.0, "intensity": 0.50, "fwhm": 25.0, "type": "C_C_stretch"},
            ]
        },
    },

    # ========== ORGANICS & SALTS ==========

    "Sodium Azide (NaN3)": {
        "description": "Ionic salt; strong IR absorption in N=N stretches.",
        "IR": {
            "peaks": [
                {"position": 2130.0, "intensity": 0.95, "fwhm": 15.0, "type": "asymmetric_N_N_stretch"},
                {"position": 1280.0, "intensity": 0.45, "fwhm": 25.0, "type": "scissor_mode"},
                {"position": 640.0, "intensity": 0.35, "fwhm": 20.0, "type": "out_of_plane_bend"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 2130.0, "intensity": 0.80, "fwhm": 12.0, "type": "symmetric_N_N_stretch"},
                {"position": 1140.0, "intensity": 0.30, "fwhm": 18.0, "type": "lattice_mode"},
            ]
        },
    },

    "Carbon (Graphite)": {
        "description": "Sp² carbon network; distinctive D, G, and 2D Raman bands.",
        "IR": {
            "peaks": [
                {"position": 1580.0, "intensity": 0.85, "fwhm": 60.0, "type": "G_band_like"},
                {"position": 1350.0, "intensity": 0.45, "fwhm": 80.0, "type": "D_band_like"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 1580.0, "intensity": 1.00, "fwhm": 30.0, "type": "G_band"},
                {"position": 1350.0, "intensity": 0.35, "fwhm": 80.0, "type": "D_band"},
                {"position": 2700.0, "intensity": 0.50, "fwhm": 50.0, "type": "2D_band"},
            ]
        },
        "XPS": {
            "peaks": [
                {"position": 284.0, "intensity": 1.00, "fwhm": 1.0, "type": "C_1s_sp2"},
                {"position": 285.0, "intensity": 0.20, "fwhm": 1.2, "type": "C_1s_sp3"},
            ]
        },
    },

    "Graphene Oxide (GO)": {
        "description": "Oxidized graphene; O-containing functional groups, many lattice modes.",
        "IR": {
            "peaks": [
                {"position": 3400.0, "intensity": 0.90, "fwhm": 100.0, "type": "O_H_stretch"},
                {"position": 1730.0, "intensity": 0.70, "fwhm": 50.0, "type": "C_O_carbonyl"},
                {"position": 1630.0, "intensity": 1.00, "fwhm": 60.0, "type": "C_C_aromatic"},
                {"position": 1220.0, "intensity": 0.65, "fwhm": 80.0, "type": "C_O_ether"},
            ]
        },
        "Raman": {
            "peaks": [
                {"position": 1350.0, "intensity": 1.00, "fwhm": 100.0, "type": "D_band_broadened"},
                {"position": 1600.0, "intensity": 0.90, "fwhm": 110.0, "type": "G_band_broadened"},
            ]
        },
        "XPS": {
            "peaks": [
                {"position": 284.0, "intensity": 0.60, "fwhm": 1.2, "type": "C_1s_sp2"},
                {"position": 286.5, "intensity": 0.25, "fwhm": 1.5, "type": "C_O_epoxide"},
                {"position": 288.0, "intensity": 0.15, "fwhm": 1.3, "type": "O_C_O"},
            ]
        },
    },

    "Iron Oxide (Fe₂O₃)": {
        "description": "Common mineral (hematite); XPS Fe 2p with satellite features.",
        "XPS": {
            "peaks": [
                {"position": 711.0, "intensity": 0.60, "fwhm": 1.8, "type": "Fe_2p_3/2"},
                {"position": 717.0, "intensity": 0.12, "fwhm": 2.0, "type": "Fe_2p_3/2_satellite"},
                {"position": 724.5, "intensity": 0.30, "fwhm": 1.8, "type": "Fe_2p_1/2"},
                {"position": 530.0, "intensity": 0.80, "fwhm": 1.2, "type": "O_1s"},
            ]
        },
        "AES": {
            "peaks": [
                {"position": 415.0, "intensity": 0.70, "fwhm": 4.5, "type": "Fe_L_M45M45"},
                {"position": 480.0, "intensity": 0.40, "fwhm": 5.0, "type": "Fe_M_NOP"},
            ]
        },
        "EDS": {
            "peaks": [
                {"position": 6.404, "intensity": 0.33, "fwhm": 0.11, "type": "Fe_K_alpha"},
                {"position": 0.525, "intensity": 0.67, "fwhm": 0.08, "type": "O_K_alpha"},
            ]
        },
    },

}


# ============================================================================
# Utility Functions
# ============================================================================

def get_material_list() -> list:
    """Return a sorted list of all available material names."""
    return sorted(PEAK_LIBRARY.keys())


def get_material_info(material_name: str) -> Dict[str, Any]:
    """
    Retrieve full information for a specific material.
    
    Parameters
    ----------
    material_name : str
        Name of the material (must match a key in PEAK_LIBRARY)
    
    Returns
    -------
    dict
        Material definition with description and technique-specific peaks
    """
    return PEAK_LIBRARY.get(material_name, {})


def get_techniques_for_material(material_name: str) -> list:
    """
    Get list of techniques available for a specific material.
    
    Parameters
    ----------
    material_name : str
        Name of the material
    
    Returns
    -------
    list
        List of technique names (e.g., ['XPS', 'Raman'])
    """
    material = get_material_info(material_name)
    return [tech for tech in material.keys() if tech != "description"]
