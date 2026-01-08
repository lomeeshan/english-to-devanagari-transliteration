import json
from pathlib import Path

KEY_ROMAN = "english word"
KEY_NATIVE = "native word"

def convert(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            roman = str(obj.get(KEY_ROMAN, "")).strip()
            native = str(obj.get(KEY_NATIVE, "")).strip()

            # Skip bad rows
            if not roman or not native or roman.lower() == "none" or native.lower() == "none":
                skipped += 1
                continue

            # Write TSV: roman<TAB>devanagari
            fout.write(f"{roman}\t{native}\n")
            written += 1

    print(f"[OK] {in_path} -> {out_path} | written={written} skipped={skipped}")

def main():
    raw_dir = Path("data/raw/hin")
    out_dir = Path("data/raw")

    mapping = [
        ("hin_train.json", "hi_train.tsv"),
        ("hin_valid.json", "hi_valid.tsv"),
        ("hin_test.json",  "hi_test.tsv"),
    ]

    for in_name, out_name in mapping:
        convert(raw_dir / in_name, out_dir / out_name)

if __name__ == "__main__":
    main()
