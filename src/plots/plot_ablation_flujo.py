#!/usr/bin/env python3
"""
Figures of the selection-flow ablation: kept as an entry point, the code lives in
src/plots/plot_ablation_compare.py (--comparison flujo).
Output: results/ablation/flujo/plots/

    PYTHONPATH=. python src/plots/plot_ablation_flujo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.plots.plot_ablation_compare import main  # noqa: E402

if __name__ == "__main__":
    main(["--comparison", "flujo", *sys.argv[1:]])
