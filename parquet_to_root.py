import hepconvert
from pathlib import Path

def convert_all_parquet_to_root():
    for p in Path('.').rglob('*.parquet'):
        if str(p).startswith("conda"):
            print(f"Skipping {p}")
            continue
        out = p.with_suffix('.root')
        print(f"Converting {p} -> {out}")
        hepconvert.parquet_to_root(str(out), str(p))

if __name__ == "__main__":
    convert_all_parquet_to_root()