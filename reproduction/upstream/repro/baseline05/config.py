from __future__ import annotations

"""Frozen Baseline05 contract.

Baseline05 inherits the frozen Baseline04 seven-member parent and runs three
non-inheriting experiments:

* E1 replaces the recent-4 ``cs_rank`` with a full-history LGBM ``cs_rank_long``.
* E2 replaces it with a recent-4 weighted linear ``cs_rank_linear``.
* E3 keeps ``cs_rank`` and *appends* E1's ``cs_rank_long`` as an eighth member.

Note the gotcha: Baseline05 ``PARENT_MEMBERS`` is the SEVEN-tuple (equal to
Baseline04's ``CHILD_MEMBERS``), distinct from Baseline04's six-tuple
``PARENT_MEMBERS``.
"""


PARENT_MEMBERS = (
    "final",
    "L88",
    "model_c",
    "level_pred",
    "rolling_v1",
    "cs_rank",
    "model_c_long",
)
# E1/E2 replace the sixth member (cs_rank) and keep seven members.
E1_MEMBERS = (*PARENT_MEMBERS[:5], "cs_rank_long", "model_c_long")
E2_MEMBERS = (*PARENT_MEMBERS[:5], "cs_rank_linear", "model_c_long")
# E3 appends cs_rank_long before model_c_long (eight members, replaces nothing).
E3_MEMBERS = (*PARENT_MEMBERS[:6], "cs_rank_long", "model_c_long")

FOLDS = (6, 7, 8)
RIDGE_ALPHA = 0.01
TARGET_STD = 0.046
EXPECTED_ROWS = 3_217_458

# E2 weighted-linear ridge ratio grid; larger ratio wins ties within 1e-12.
LINEAR_RATIO_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

# CS-rank training windows shared by E1/E2 (see rank_data.training_partitions).
CS_RANK_WINDOW = 4

BASE_OOF_FILENAMES = {
    6: "fold6_five_member_oof.npz",
    7: "fold7_five_member_oof.npz",
    8: "fold8_five_member_oof.npz",
}

# --- Frozen Baseline04 anchor hashes (from Frozen Baseline04 Contract) -------
# Base five-member OOF MD5s carried forward from Baseline04.
EXPECTED_BASE_OOF_MD5 = {
    6: "ec0b39bcb9e463e8d93384d300f432c5",
    7: "112578112fdfc09430a37658013bd565",
    8: "bdfdd54eeb64035c932662a9df8dba09",
}
# SHA256 of frozen parent artifacts that must not change before any child runs.
EXPECTED_SHA256 = {
    "parent_submission": "08eda4dde422821cabcc7bb39289b7cb3e974880ebb369e91056a2cde9013db2",
    "public_members": "5525f091f0cb785b1b6ff6ee2223573becff1ab97df6046a70a44b1e4cf67c52",
    "parent_recipe": "68761ef514d03444d590495928f5591217b67efd70d2d66ba32b9503abb2b3a3",
    "parent_report": "9ac57e400685d4ee60f8740d5b81719033c4aa172747153f89ebedb15f541085",
    # model_c_long full-history OOF + public parquet carried from Baseline03/04.
    "long_fold_6": "bba9f540df15c0864eae4e2e602b2456a0a3c0e33da05cb2f51752b7461edc79",
    "long_fold_7": "9ec7218e9665a6b05b6fa3cc7561a532caebc1b43412e92426ee0ac3ab95012b",
    "long_fold_8": "199b17d42fec7005a45fe40511f51963f7d936242d3faf9c4a536045ac375e8e",
    "public_model_c_long": "022340e324dd2e2c81da8b90466d284080cdb8ff03097636b869c4de68007f1f",
}
