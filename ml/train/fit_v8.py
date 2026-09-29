"""Derive and verify the stage-2 FIT-v8 manifest from the frozen FIT split (stdlib only).

FIT-v8 is the original FIT role of episodes/v1 (5,746 rows, 928 identities) minus 24 rows
whose target boxes were rejected by visual review in filters v2..v8. Every retained CSV line
is byte-identical to the organizer's dataset/train.csv; v8 is written with CRLF line ends,
exactly as the historical filter chain wrote it.
"""

import argparse
import csv
import hashlib
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIT_CSV = HERE / "data/episodes_v1/train.csv"
FIT_V8_CSV = HERE / "data/fit_v8/train.csv"
ORGANIZER_TRAIN_CSV = HERE.parents[1] / "dataset/train.csv"

FIT_SHA256 = "e900f4cda82a0532e407b371bde180d2aa466677bf8e87548bee82305806f434"
FIT_V8_SHA256 = "fe1394a9fbc7ba5e21eeed802252d373348940ebece6d0ae5a6c4727215ff962"
ORGANIZER_TRAIN_SHA256 = "bd1df45b052ae9aabb7fd356898e244875f8cad2f111a3f76977c1b90bce9268"

FIELDS = ("image_id", "x", "y", "w", "h", "vehicle_id", "camera_id")
# Exact exclusions per filter version, copied from verify_fitting_input() of the historical
# runner (git show 058c348^:ml/scripts/run_experiment.py). Keys are full CSV rows.
EXCLUSIONS = {
    "v2": [("2c29c30a0883406aad3d981375c83b86", "0", "254", "368", "474", "193", "43")],
    "v3": [
        ("7bfca3ec2ad34b07ae3fb4296534a32a", "225", "254", "677", "448", "93", "43"),
        ("8aa7acf1d8334ace802bb423682e1951", "1", "147", "878", "510", "93", "43"),
        ("abb9a1c3298743d9ab825c981ca513e4", "742", "904", "716", "174", "1192", "90"),
    ],
    "v4": [
        ("065e9ef1b9bb483795b5cba866976bd9", "586", "546", "581", "357", "51", "65"),
        ("1c0d8ac1cee94629b1abb8e2e2a026f5", "0", "283", "779", "435", "1384", "90"),
        ("5ef3ef2c31834898acb317e1a7d4b058", "583", "559", "583", "350", "51", "65"),
        ("6a35d140d5b3443082983ddb92385946", "1154", "780", "765", "300", "714", "39"),
        ("6bd0ca46f9334ef5a3a1fbe01c797af0", "1042", "496", "325", "370", "1481", "15"),
        ("8550c35a89f941e6b128b44eca26efcf", "803", "418", "600", "407", "1083", "51"),
        ("b5797257de1f466ca3328cf6dac53c4b", "0", "282", "799", "436", "1384", "90"),
        ("dd73db30d78a43cd9f13680a6572ec78", "782", "418", "593", "402", "1083", "51"),
        ("fdb84b0a7c7f44f2b3787258c796e511", "0", "288", "780", "434", "1384", "90"),
    ],
    "v5": [
        ("4d4114ce60a341218220ae99db9c76cc", "802", "645", "460", "350", "879", "63"),
        ("b1f8016a14154795b748690ab650f8fd", "1361", "577", "557", "366", "879", "90"),
    ],
    "v6": [
        ("17327c0f237a46d1ad076a59fb234f7d", "0", "307", "379", "411", "1276", "90"),
        ("2a1deae41c354f0a9f959b75e0e5e556", "913", "497", "1005", "528", "1276", "89"),
        ("31bc8ee6b49646fe8533ac1628887e94", "1", "78", "1151", "614", "509", "43"),
        ("6d56e158b7be47029609c56d531f2e86", "133", "351", "628", "408", "757", "90"),
        ("975d95889ce644f7b021c54bdb89326d", "554", "647", "984", "433", "849", "43"),
        ("b7fde089f17a48548c982909a68cafb4", "383", "86", "766", "629", "1274", "68"),
    ],
    "v7": [("ec14ef22e2704372a12d94b81370589b", "603", "572", "695", "449", "1138", "19")],
    "v8": [
        ("2223367bff1d4237a34212ee3df0f0df", "728", "627", "753", "453", "967", "90"),
        ("8f28abd7a6104e9d84068eb01ea04363", "728", "625", "753", "455", "967", "90"),
    ],
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def rows_with_lines(data):
    """Parsed rows paired with their original byte lines (header excluded)."""
    lines = data.splitlines()
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if len(rows) != len(lines) - 1:
        raise ValueError("CSV rows must be one per physical line")
    return rows, lines


def derive_fit_v8(fit_bytes):
    """Remove exactly the 24 reviewed rows, keep order and original line bytes, write CRLF."""
    rows, lines = rows_with_lines(fit_bytes)
    excluded = {key for keys in EXCLUSIONS.values() for key in keys}
    keys = [tuple(row[field] for field in FIELDS) for row in rows]
    missing = excluded - set(keys)
    if missing or len(excluded) != 24:
        raise ValueError(f"FIT split does not contain every reviewed exclusion: {sorted(missing)}")
    kept = [line for key, line in zip(keys, lines[1:], strict=True) if key not in excluded]
    if len(kept) != len(rows) - 24:
        raise ValueError("Each reviewed exclusion must match exactly one FIT row")
    return b"".join(line + b"\r\n" for line in [lines[0], *kept])


def check(organizer_csv=None):
    fit = FIT_CSV.read_bytes()
    if sha256_bytes(fit) != FIT_SHA256:
        raise ValueError("Frozen FIT split changed")
    derived = derive_fit_v8(fit)
    if sha256_bytes(derived) != FIT_V8_SHA256 or FIT_V8_CSV.read_bytes() != derived:
        raise ValueError("FIT-v8 is not the exact documented derivation of the FIT split")
    report = {"fit_rows": len(fit.splitlines()) - 1, "fit_v8_rows": len(derived.splitlines()) - 1,
              "fit_v8_identities": len({row["vehicle_id"] for row in rows_with_lines(derived)[0]}),
              "fit_v8_sha256": FIT_V8_SHA256}
    if organizer_csv is not None:
        source = Path(organizer_csv).read_bytes()
        if sha256_bytes(source) != ORGANIZER_TRAIN_SHA256:
            raise ValueError("Organizer train.csv differs from the pinned contest release")
        source_lines = source.splitlines()
        if source_lines[0] != fit.splitlines()[0] or not set(fit.splitlines()[1:]) <= set(source_lines[1:]):
            raise ValueError("FIT rows are not byte-identical organizer train.csv rows")
        report["organizer_train_sha256"] = ORGANIZER_TRAIN_SHA256
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organizer-csv", type=Path, default=ORGANIZER_TRAIN_CSV,
                        help="Also prove FIT rows come from this organizer train.csv (skipped if absent)")
    args = parser.parse_args()
    organizer = args.organizer_csv if args.organizer_csv.is_file() else None
    print(check(organizer))


if __name__ == "__main__":
    main()
