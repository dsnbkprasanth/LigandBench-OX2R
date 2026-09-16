from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_dataset_counts():
    df = pd.read_csv(ROOT / 'data' / 'ox2r_curated_6415.csv')
    assert len(df) == 6415
    assert int(df['label'].sum()) == 747
    assert int((df['label'] == 0).sum()) == 5668
    assert df['molecule_chembl_id'].is_unique
    assert df['canonical_smiles'].notna().all()


def test_reproducibility_config():
    text = (ROOT / 'config' / 'reproducibility.yaml').read_text()
    assert 'chembl_release: 34' in text
    assert 'seed: 42' in text
    assert 'cluster_count: 120' in text
    assert 'similarity_bootstrap: 2000' in text


def test_required_scripts_exist():
    for name in ['benchmark_frozen.py', 'similarity_stratified.py', 'cluster_split.py', 'reproduce_main.py']:
        assert (ROOT / 'scripts' / name).exists()
