from __future__ import annotations


PARENT_MEMBERS = (
    "final",
    "L88",
    "model_c",
    "level_pred",
    "rolling_v1",
    "cs_rank",
)
CHILD_MEMBERS = (*PARENT_MEMBERS, "model_c_long")
FOLDS = (6, 7, 8)
RIDGE_ALPHA = 0.01
TARGET_STD = 0.046
EXPECTED_ROWS = 3_217_458

BASE_OOF_FILENAMES = {
    6: "fold6_five_member_oof.npz",
    7: "fold7_five_member_oof.npz",
    8: "fold8_five_member_oof.npz",
}
EXPECTED_BASE_OOF_MD5 = {
    6: "ec0b39bcb9e463e8d93384d300f432c5",
    7: "112578112fdfc09430a37658013bd565",
    8: "bdfdd54eeb64035c932662a9df8dba09",
}
EXPECTED_SHA256 = {
    "parent_recipe": "18935931c880d258a778686fa7077778956aad75729f8c1dac663a1d6add2bac",
    "parent_submission": "057791cbb6e280691525f46cd529c8ecd8e58967e704e3792b7046e20ee0b1c1",
    "public_members": "635ec1cafa769b70759e4de63b1391dba497a34f786152c9977fc31525c1fc97",
    "long_fold_6": "bba9f540df15c0864eae4e2e602b2456a0a3c0e33da05cb2f51752b7461edc79",
    "long_fold_7": "9ec7218e9665a6b05b6fa3cc7561a532caebc1b43412e92426ee0ac3ab95012b",
    "long_fold_8": "199b17d42fec7005a45fe40511f51963f7d936242d3faf9c4a536045ac375e8e",
    "public_model_c_long": "022340e324dd2e2c81da8b90466d284080cdb8ff03097636b869c4de68007f1f",
}
