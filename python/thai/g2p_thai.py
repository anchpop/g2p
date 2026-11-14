"""Thai G2P server for the g2p crate.

Reads one JSON request per line on stdin — {"text": "..."} — and writes one
JSON response per line: {"ipa": "...", "words": [[surface, ipa], ...]}, with
vachana-thai's IPA string exactly as `th2ipa` returns it (words separated by
spaces, tone diacritics on the first vowel character of each syllable), or
{"error": "..."}. Surfaces come from the same conversion loop as the IPA.
`source_spans` carries parallel original UTF-8 byte ranges (null for separators),
or an empty list when the entire sequence cannot be traced exactly.

A request that also carries the caller's segmentation — {"text": "...",
"words": ["...", ...]} — is answered with {"word_ipa": ["...", ...]}, one IPA
string per given word ("" for an unspoken one): each word is converted on its
own, without pythainlp's segmenter.

The crate tokenizes and labels that string itself (src/thai.rs); this process only
runs the parts that live in Python.

`identity` prints the resolved package versions so the crate can stamp its
output with them.
"""

import json
import re
import sys
import unicodedata
from importlib.metadata import version


def source_spans(text, words):
    """Trace the native repetition tokens through its second segmentation.

    Token-local normalization is checked using the backend's own transforms.
    Repetition ownership comes from the same tokenizer expand_maiyamok uses,
    not from guessing which final phoneme word was repeated.
    """
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import expand_maiyamok, normalize
    from vachana_g2p.vachana_g2p import clean_text, number_to_text

    def prepare(value):
        return normalize(number_to_text(clean_text(value).replace(" ๆ", "ๆ ")))

    prepared = prepare(text)
    tokens = [part for token in word_tokenize(prepared)
              for part in re.split("(ๆ)", token) if part]
    expanded = []
    cursor = 0
    for token in tokens:
        if token.isspace():
            expanded.append([token, None])
            continue
        while cursor < len(text) and (text[cursor].isspace()
                or unicodedata.category(text[cursor]).startswith("P")):
            cursor += 1
        start = cursor
        # Apply the backend's own deterministic normalization to source pieces.
        # A piece is accepted only if it produces the entire native token exactly;
        # a rewrite spanning token boundaries is withheld, never fuzzy-matched.
        # Normalization only drops a few marks and number expansion shortens the
        # source, so a bounded window keeps this linear in the text length.
        ends = []
        for end in range(cursor + 1, min(len(text), cursor + len(token) + 8) + 1):
            piece = text[cursor:end]
            if piece[-1].isspace() or ("ๆ" in piece and piece != "ๆ"):
                break
            if piece == token.strip() or prepare(piece) == token.strip():
                ends.append(end)
        if not ends:
            return []
        cursor = ends[-1]
        if token == "ๆ":
            while expanded and expanded[-1][0].isspace():
                expanded.pop()
            if not expanded:
                return []
            base, owner = expanded[-1]
            # Every copy, including the original, owns base plus repetition mark.
            for entry in expanded:
                if entry[1] == owner:
                    entry[1] = (owner[0], cursor)
            expanded.append([base, (owner[0], cursor)])
        else:
            expanded.append([token, (start, cursor)])
    if any(not (c.isspace() or unicodedata.category(c).startswith("P"))
           for c in text[cursor:]):
        return []
    if "".join(t for t, _ in expanded) != "".join(expand_maiyamok(prepared)):
        return []
    # g2p removes token-internal spaces. Preserve the owner of every other char.
    stream = [(c, owner) for token, owner in expanded for c in token if not c.isspace()]
    at = 0
    spans = []
    for surface, _ in words:
        if not surface:
            spans.append(None)
            continue
        part = stream[at:at + len(surface)]
        if "".join(c for c, _ in part) != surface or any(owner is None for _, owner in part):
            return []
        start = min(owner[0] for _, owner in part)
        end = max(owner[1] for _, owner in part)
        spans.append((len(text[:start].encode()), len(text[:end].encode())))
        at += len(surface)
    return spans if at == len(stream) else []


def word_ipa(words):
    """IPA for each of the caller's words, converted as th2ipa converts each
    token of its own segmentation: the same text preparation (word_chunk
    without its tokenizer), then vachana's dictionary or TLTK for the word.
    Cleaning can split a word (punctuation becomes a space); ๆ repeats the
    piece before it, as expand_maiyamok repeats the token before it."""
    from pythainlp.util import normalize
    from vachana_g2p.vachana_g2p import (
        THAI2PHONE_DICT, clean_text, decode, g2p, number_to_text, validate, word_chunk)

    def piece_ipa(piece):
        tokens = g2p([piece], decoded=False, return_tokens=True)
        # TLTK can leave part of a long unknown word untranscribed (ดีพโรส
        # as ดี|พโรส with only dii0): the syllables it returns do not
        # validate, and decoding them would silently drop the rest. Such a
        # word is read as vachana's own segmenter splits it instead.
        if piece not in THAI2PHONE_DICT and not all(validate(phone) for _, phone in tokens):
            tokens = g2p(word_chunk(piece), decoded=False, return_tokens=True)
        return [decode(phone, "ipa") for _, phone in tokens]

    result = []
    previous = []
    for word in words:
        ipa = []
        prepared = normalize(number_to_text(clean_text(word)))
        for piece in re.split(r"\s+|(ๆ)", prepared):
            if not piece:
                continue
            if piece != "ๆ":
                previous = piece_ipa(piece)
            ipa.extend(previous)
        result.append(" ".join(ipa))
    return result


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "identity":
        print(f"vachana-g2p/{version('vachana-g2p')} pythainlp/{version('pythainlp')}")
        return
    from vachana_g2p.vachana_g2p import clean_text, g2p, word_chunk

    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if "words" in request:
                out.write(json.dumps({"word_ipa": word_ipa(request["words"])}, ensure_ascii=False) + "\n")
                out.flush()
                continue
            text = request["text"]
            # The exact loop used by th2ipa, retaining each emitted token's
            # surface rather than independently re-tokenizing the input.
            words = g2p(word_chunk(clean_text(text)), transcription="ipa", return_tokens=True)
            ipa = " ".join(phone for _, phone in words).replace("  ", ", ")
            try:
                spans = source_spans(text, words)
            except Exception:  # Attribution must never discard a pronunciation.
                spans = []
            out.write(json.dumps({"ipa": ipa, "words": words, "source_spans": spans}, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — every failure must answer the request
            out.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
