# Frozen upstream source snapshot

This directory contains the source-only snapshot of the team baseline pipeline
at commit `0ad2c976d02ff3fc2d1965b5cb0a484800503470`.

It is vendored here because the original experimental branches contained large
OOF arrays, public member predictions, model files, and submissions that should
not be exposed in a portfolio repository. Those generated artifacts are not
included.

For a full rebuild, copy this directory to a writable runtime location, produce
the five-member OOF and public member artifacts from the official competition
data, and set `TEAMMATE_WT` to that runtime location. The expected layout is:

```text
TEAMMATE_WT/
├── UPSTREAM_COMMIT
├── repro/
└── artifacts/
    ├── oof/
    │   ├── fold6_five_member_oof.npz
    │   ├── fold7_five_member_oof.npz
    │   └── fold8_five_member_oof.npz
    └── member_predictions/
        ├── test_metadata.csv
        ├── public_final_fallback.csv
        ├── public_model_c.csv
        ├── public_level_pred.csv
        └── public_rolling_v1.csv
```

The frozen artifacts were retained in a private archival bundle for team
reproduction. They are intentionally excluded from the public repository.
