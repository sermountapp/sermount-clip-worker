#!/usr/bin/env python3
"""
Copied from sermount-web (same checker, this repo's exclude list).

Compare a detect-secrets scan result against the committed baseline.

Exits non-zero if any new secrets (not in baseline) are found.
Prints only finding type and filename, never the actual secret value.

Usage (from repo root):
    detect-secrets scan ... > /tmp/scan-results.json
    python3 .github/scripts/check-secrets-baseline.py
"""

import json
import sys

BASELINE_PATH = ".secrets.baseline"
SCAN_RESULTS_PATH = "/tmp/scan-results.json"
EXCLUDE_ARGS = (
    "--exclude-files '\\.secrets\\.baseline' "
    "--exclude-files '__pycache__/.*' "
    "--exclude-files '\\.venv/.*' "
    "--exclude-files 'out/.*'"
)


def load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: could not parse {path}: {e}")
        sys.exit(1)


def finding_key(filename: str, finding: dict) -> tuple[str, str, str]:
    return (
        filename,
        finding.get("type", ""),
        finding.get("hashed_secret", ""),
    )


def main() -> None:
    baseline = load_json(BASELINE_PATH)
    current = load_json(SCAN_RESULTS_PATH)

    baseline_findings = {
        finding_key(filename, finding)
        for filename, findings in baseline.get("results", {}).items()
        for finding in findings
    }

    new_findings = []
    for filename, findings in current.get("results", {}).items():
        for finding in findings:
            if finding_key(filename, finding) not in baseline_findings:
                new_findings.append(f"{filename}: {finding['type']}")

    if new_findings:
        print("Potential secrets not in baseline:")
        for info in new_findings:
            print(f"  {info}")
        print()
        print("If these are confirmed false positives, regenerate the baseline:")
        print(f"  detect-secrets scan {EXCLUDE_ARGS} > .secrets.baseline")
        print()
        print("Then review the baseline with: detect-secrets audit .secrets.baseline")
        sys.exit(1)

    total = sum(len(v) for v in current.get("results", {}).values())
    print(
        f"Secret scan passed — {total} finding(s) in baseline, no new secrets detected."
    )


if __name__ == "__main__":
    main()
