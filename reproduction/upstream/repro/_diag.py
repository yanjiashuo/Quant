import sys
from pathlib import Path
base = Path(r"C:\Users\LENOVO\Desktop\Quant\artifacts\models\final_deploy_package")
for d in ["model_a_000008","model_b","model_c","weight_model","level_model"]:
    p = base / d / "lgb_model.txt"
    raw = p.read_bytes()
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n")
    has_cr = b"\r" in raw
    print(f"{d}: bytes={len(raw)} CRLF={crlf} LF_total={lf} has_CR={has_cr}")
