"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 13 (symmetrisch, Zeilensumme 1, Eigenwerte 1, 0,5, 0,1 zu den Vektoren (1,1,1), (1,0,-1), (1,-2,1)), b = (1,3, 1,4, 1,3), Wahrheit x = (1, 2, 1) (|x|² = 6), t = 1.
    Handrechnung:
    - Skalar-Netz K = 1, gamma = 1, theta = 0,1: z = gamma A b = (1,33, 1,34, 1,33), x1 = soft(z, 0,1) = (1,23, 1,24, 1,23) (gleich dem Schritt aus Stück 13); Verlust |x1 - x|²/|x|² = 0,6834/6 = 0,1139;
    - Gradienten (alle z über der Schwelle, Maske 1): dL/dtheta = -2 Σ(x1 - x)/|x|² = +0,1; dL/dgamma = 2 Σ(x1 - x)(A b)/|x|² = -0,135533;
    - LISTA K = 1 (We = A^T, theta = 0,1) hat dieselbe Ausgabe; dL/dWe = dz^T b mit dz = 2 (x1 - x)/6 = (0,076667, -0,253333, 0,076667), dL/dWe[0, 0] = 0,099667; dL/dS = 0 (Start 0);
    - ISTA mit zwei Schritten (lambda = 0,1): x2 = soft(x1 - A (A x1 - b), 0,1) = (1,2267, 1,2466, 1,2267) mit A x1 = (1,233, 1,234, 1,233);
    - Tikhonov mit lambda = 0,1 und Posterior-Mittel (P = I, sigma = 0,1): (1,153465, 1,653465, 1,153465)."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    b = np.array([[1.3, 1.4, 1.3]])
    x = np.array([[1.0, 2.0, 1.0]])
    return a, b, x
