"""
LigandBench
===========

A target-agnostic, scaffold-aware ligand-based machine-learning benchmarking
pipeline. It reproduces the workflow of Drewe (J. Cheminformatics, 2026) for
ADRB3 agonists, but generalized so it runs for *any* target:

    ChEMBL fetch  ->  consensus labeling  ->  RDKit curation  ->
    featurization (ECFP4 / ECFP6 / RDKit descriptors / hybrids)  ->
    scaffold-aware split + grouped-CV benchmark (6 classifiers)  ->
    Y-randomization  ->  SHAP  ->  applicability domain.

Each stage lives in its own module and can be used independently or driven
end-to-end through ``pipeline.run_pipeline`` or the Streamlit GUI (``app.py``).
"""

__version__ = "0.1.0"
