"""g2pk2 tags with python-mecab-ko (`import mecab; mecab.MeCab().pos(text)`),
which builds mecab from source at install time and fails on systems without
that toolchain (it does not build under nix). This module stands in for it,
backed by the prebuilt `mecab-ko` wheel and its `mecab-ko-dic` dictionary,
which is the same analyzer and dictionary python-mecab-ko would have built.
Only `pos` is used (see g2pk2.utils.annotate)."""

import mecab_ko


class MeCab:
    def __init__(self) -> None:
        self._tagger = mecab_ko.Tagger()

    def pos(self, text: str) -> list[tuple[str, str]]:
        out = []
        for line in self._tagger.parse(text).splitlines():
            if line == "EOS" or "\t" not in line:
                continue
            surface, features = line.split("\t", 1)
            out.append((surface, features.split(",")[0]))
        return out
