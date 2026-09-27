"""Build per-country TRAIN blocking index for diagnostics (production config)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rushi/er/student_resource/code/business_entity_resolution")
from src.submit import open_or_build_index

ROOT = Path("/home/rushi/er/student_resource")
DIAG = ROOT / "output" / "diagnostic_v1"

country = sys.argv[1].strip().lower()
index_path = DIAG / f"train_blocking_{country}.sqlite"
t0 = time.time()
index = open_or_build_index(
    [ROOT / "dataset/train/train_source2.tsv", ROOT / "dataset/train/train_source3.tsv"],
    index_path=index_path,
    target_limit=None,
    bucket_limit=5000,
    country=country,
)
try:
    meta = index.metadata() if hasattr(index, "metadata") else {}
finally:
    if hasattr(index, "close"):
        index.close()
print(f"BUILT country={country} path={index_path} elapsed={time.time()-t0:.1f}s metadata={meta}", flush=True)
