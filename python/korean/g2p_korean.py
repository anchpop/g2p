"""Korean G2P server for the g2p crate.

Reads one JSON request per line on stdin — {"clauses": [["...", ...], ...]},
one utterance as clauses (runs of Hangul words between punctuation marks),
each a list of space-delimited Hangul words (어절), Hangul syllables only —
and writes one JSON response per line: {"prons": [[...], ...]} with each
word's pronunciation as post-sandhi Hangul (g2pk's output form: 값이 → 갑씨,
꽃잎 → 꼰닙), same shape as the request, or {"error": "..."}. The crate maps
that Hangul to phones itself (src/korean.rs); this process only runs the
parts that live in Python.

Each clause goes through g2pk's pipeline as one string, so its sound-change
rules apply across the spaces inside the clause: 못 만났어 → [몬만나써],
부엌 좀 → [부억쫌], 할 것 → [할껏]. That is what connected speech contains —
a speaker does not pause between 어절 — and the labels target real audio.
Punctuation is where speakers do pause, so a clause boundary blocks the
rules: 안녕, 라디오 stays [안녕 라디오] rather than becoming [나디오].
Morphological tagging (mecab-ko) sees the whole clause too.

g2pk2 transliterates English words via CMUdict, which it downloads through
nltk at import time. The crate never sends Latin text (it refuses it), so
the download is stubbed out here rather than fetched. Its number-reading
and idiom steps are likewise never reached by digits (refused), and the
idiom table is applied once per clause.

`identity` prints the resolved package versions so the crate can stamp its
output with them.
"""

import contextlib
import json
import re
import sys
import types
from importlib.metadata import version


def _stub_nltk() -> None:
    nltk = types.ModuleType("nltk")
    nltk.data = types.SimpleNamespace(find=lambda *a, **k: None)
    nltk.download = lambda *a, **k: None
    corpus = types.ModuleType("nltk.corpus")
    corpus.cmudict = types.SimpleNamespace(dict=lambda: {})
    nltk.corpus = corpus
    sys.modules["nltk"] = nltk
    sys.modules["nltk.corpus"] = corpus


class Phonemizer:
    def __init__(self) -> None:
        _stub_nltk()
        import g2pk2  # noqa: F401 — package import wires the submodules
        from g2pk2 import special, regular, utils
        from jamo import h2j

        self.g2p = g2pk2.G2p()  # loads mecab (via mecab.py here) and table.csv
        self.h2j = h2j
        self.annotate = utils.annotate
        self.compose = utils.compose
        self.special = (
            special.jyeo, special.ye, special.consonant_ui, special.josa_ui,
            special.vowel_ui, special.jamo, special.rieulgiyeok, special.rieulbieub,
            special.verb_nieun, special.balb, special.palatalize, special.modifying_rieul,
        )
        self.link = (regular.link1, regular.link2, regular.link3, regular.link4)
        # idioms.txt is re-read on every G2p.idioms call; load it once.
        self.idioms: list[tuple[str, str]] = []
        with open(self.g2p.idioms_path, encoding="utf8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if "===" in line:
                    a, b = line.split("===")
                    self.idioms.append((a, b))

    def clause(self, words: list[str]) -> list[str]:
        """g2pk's steps 1 and 3–9 (English and number conversion skipped) on
        the clause as one string; the output keeps one token per word."""
        text = " ".join(words)
        for a, b in self.idioms:
            text = re.sub(a, b, text)
        text = self.annotate(text, self.g2p.mecab)
        inp = self.h2j(text)
        for func in self.special:
            inp = func(inp, False, False)
        inp = re.sub("/[PJEB]", "", inp)
        for str1, str2, _ in self.g2p.table:
            inp = re.sub(str1, str2, inp)
        for func in self.link:
            inp = func(inp, False, False)
        out = self.compose(inp).split(" ")
        if len(out) != len(words):
            raise ValueError(f"word count changed: {words!r} -> {out!r}")
        return out

    def prons(self, clauses: list[list[str]]) -> list[list[str]]:
        return [self.clause(words) for words in clauses]


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "identity":
        print(
            f"g2pk2/{version('g2pk2')} mecab-ko/{version('mecab-ko')} "
            f"mecab-ko-dic/{version('mecab-ko-dic')} jamo/{version('jamo')}"
        )
        return
    # g2pk2 prints progress ("mecab installed") to stdout while loading; keep
    # the JSON stream clean.
    with contextlib.redirect_stdout(sys.stderr):
        phonemizer = Phonemizer()
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            clauses = json.loads(line)["clauses"]
            out.write(json.dumps({"prons": phonemizer.prons(clauses)}, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — every failure must answer the request
            out.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
