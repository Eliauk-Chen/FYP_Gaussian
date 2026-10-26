"""Check prerequisites without loading private data, fitting models or downloading files."""
from pathlib import Path
import argparse
from hardware_paths import HARDWARE_ROOT, RESPONSE_BANK

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['clean','perturbations','width','multiwidth','cycle'],default='clean')
    stage=parser.parse_args().stage
    files=[HARDWARE_ROOT/'src/rbf_hardware/modeling/joint_gaussian.py',RESPONSE_BANK,
           Path('artifacts/hagrid_landmarks_v1/landmark_features.npz'),
           Path('artifacts/hagrid_landmarks_v1/development_manifest.csv')]
    files += [Path(f'artifacts/hagrid_landmarks_v1/fold_{f}/split_indices.json') for f in range(3)]
    if stage!='clean':
        files += [Path(f'artifacts/hagrid_hardware_clean_v2/fold_{f}/model_bundle.joblib') for f in range(3)]
        files += [Path(f'artifacts/hagrid_failure_modes_v1/corruptions_seed_{seed}.npz') for seed in range(10)]
    if stage in ['multiwidth','cycle']:
        files += [Path(f'artifacts/hagrid_hardware_width_scan_v1/fold_{f}/models.joblib') for f in range(3)]
    if stage=='cycle':
        files += [Path(f'artifacts/hagrid_hardware_multiwidth_v1/fold_{f}/{kind}/models.joblib') for f in range(3) for kind in ['jitter','missing','outlier']]
        files += [Path(f'artifacts/hagrid_hardware_multiwidth_v1/predictions_noise_seed_{seed}.npz') for seed in range(10)]
    missing=[p for p in files if not p.is_file()]
    for p in missing:print(f'MISSING: {p}')
    if missing:
        print('Supply authorized external inputs; see docs/研究设计与复现.md. No substitute data were generated.')
        raise SystemExit(1)
    print(f'Primary input files present for {stage}; this does not replace model/data alignment audits.')

if __name__=='__main__':main()
