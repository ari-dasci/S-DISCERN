"""
Ablation of the selection flow of PARAMETRIC (tests vs cv): kept as an entry point,
the analysis lives in src/analysis/ablation_compare.py (--comparison flujo).
Output in results/ablation/flujo/.

    PYTHONPATH=. python src/analysis/ablation_flujo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analysis.ablation_compare import main  # noqa: E402

if __name__ == "__main__":
    main(["--comparison", "flujo", *sys.argv[1:]])
