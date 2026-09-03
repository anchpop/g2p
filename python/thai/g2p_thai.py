"""Thai G2P server for the g2p crate.

Reads one JSON request per line on stdin — {"text": "..."} — and writes one
JSON response per line: {"ipa": "..."} with vachana-thai's IPA string exactly
as `th2ipa` returns it (words separated by spaces, tone diacritics on the
first vowel character of each syllable), or {"error": "..."}. The crate
tokenizes and labels that string itself (src/thai.rs); this process only
runs the parts that live in Python.

`identity` prints the resolved package versions so the crate can stamp its
output with them.
"""

import json
import sys
from importlib.metadata import version


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "identity":
        print(f"vachana-g2p/{version('vachana-g2p')} pythainlp/{version('pythainlp')}")
        return
    from vachana_g2p import th2ipa  # slow import; done once per process

    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line)["text"]
            out.write(json.dumps({"ipa": th2ipa(text)}, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — every failure must answer the request
            out.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
