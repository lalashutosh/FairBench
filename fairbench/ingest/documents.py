"""Filings as addressable paragraphs, and quotes as addresses.

A fund's rules come from its prospectus, summary prospectus and statement of additional
information. Every extracted rule must point back to the exact place it came from, so this
module turns a raw document into paragraphs and finds where a quoted sentence sits. It does
not interpret a rule.

    bytes --parse_html / parse_text / parse_pdf--> ParsedDocument (paragraphs + full text)
    ParsedDocument + quote --locate_quote--> Locator (paragraph(s), page, heading, offsets)

Contract
    * ``ParsedDocument.text`` is the paragraphs' text joined by "\\n", and every
      ``Paragraph`` records its span in it: ``doc.text[p.char_start:p.char_end] == p.text``.
      Paragraph text has whitespace (non-breaking spaces included) collapsed to single spaces.
    * ``parse_html`` drops script, style, head, title and inline-XBRL header content and keeps
      the text of every other tag, including ``ix:nonNumeric``. A paragraph is a heading when
      it is an h1-h6, or a short one (<= 120 characters) set entirely in bold or in capitals.
      Cells of one table row are joined with " | ". Each paragraph carries the nearest heading
      at or before it (a heading carries its own text, so ``section`` returns a section's title
      line and body, and never the next section's title). Headings are flat: a sub-heading
      replaces its parent for the paragraphs under it.
    * ``locate_quote`` matches on ``normalise`` (Unicode NFKC, quote and dash typography
      folded to ASCII, case folded, whitespace collapsed) and may span adjacent paragraphs.
      The offsets it returns are in the ORIGINAL ``doc.text``, mapped back character by
      character. It returns None unless the quote is present: no fuzzy matching, and a match
      may not begin or end inside a word or a number ("0% of" is not found in "80% of",
      "5%" is not found in "2.5%").
    * ``numbers_in`` and ``number_appears`` support the check that a threshold literally
      appears in its quoted evidence. They read digits, percentages, spelled-out numbers and
      a few fractions, and skip digits that belong to identifiers such as "12b-1" or "10-K".

Standard library only. ``parse_pdf`` imports ``pypdf`` when called and raises ImportError
naming it if it is missing.
"""
from __future__ import annotations

import bisect
import hashlib
import io
import math
import re
import unicodedata
from array import array
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterator

MEDIA_HTML, MEDIA_PDF, MEDIA_TEXT = "text/html", "application/pdf", "text/plain"
MAX_HEADING_CHARS = 120


# ---------------------------------------------------------------- data model
@dataclass(frozen=True)
class Paragraph:
    index: int
    text: str
    char_start: int                # offsets into ParsedDocument.text
    char_end: int
    heading: str | None = None     # nearest heading at or before this paragraph
    kind: str = "paragraph"        # heading | paragraph | list_item | table_row
    page: int | None = None        # 1-based for PDF, None for HTML and text


@dataclass
class ParsedDocument:
    paragraphs: list[Paragraph]
    text: str                      # "\n".join(p.text for p in paragraphs)
    sha256_of_source: str          # hex digest of the raw bytes that were parsed
    media_type: str                # text/html | application/pdf | text/plain
    _norm: _NormIndex | None = field(default=None, init=False, repr=False, compare=False)

    def section(self, heading_pattern: str) -> list[Paragraph]:
        """Paragraphs whose heading matches the regex (case-insensitive search), e.g.
        "principal investment strateg" or "fundamental (investment )?(restrictions|policies)"."""
        rx = re.compile(heading_pattern, re.IGNORECASE)
        return [p for p in self.paragraphs if p.heading is not None and rx.search(p.heading)]

    def _index(self) -> _NormIndex:
        if self._norm is None or self._norm.source is not self.text:
            self._norm = _NormIndex(self)
        return self._norm


@dataclass(frozen=True)
class Locator:
    """Where a quote sits. ``text`` is the document's own wording of the span."""
    paragraph_index: int
    paragraph_end_index: int       # equals paragraph_index unless the quote spans paragraphs
    char_start: int                # offsets into ParsedDocument.text
    char_end: int
    page: int | None
    heading: str | None
    text: str

    def as_string(self) -> str:
        i, j = self.paragraph_index, self.paragraph_end_index
        where = f"para {i}" if i == j else f"paras {i}-{j}"
        if self.page is not None:
            where = f"page {self.page}, {where}"
        if self.heading:
            where += f' under "{self.heading}"'
        return where


def _collapse(s: str) -> str:
    return " ".join(s.split())     # str.split() treats non-breaking and other Unicode spaces as blanks


def _assemble(blocks: list[tuple[str, str, int | None]], raw: bytes, media_type: str) -> ParsedDocument:
    """(text, kind, page) blocks in reading order -> paragraphs with offsets and headings."""
    paragraphs: list[Paragraph] = []
    heading: str | None = None
    pos = 0
    for text, kind, page in blocks:
        if not text:
            continue
        if kind == "heading":
            heading = text                 # a heading belongs to the section it opens
        paragraphs.append(Paragraph(len(paragraphs), text, pos, pos + len(text), heading, kind, page))
        pos += len(text) + 1
    return ParsedDocument(paragraphs, "\n".join(p.text for p in paragraphs),
                          hashlib.sha256(raw).hexdigest(), media_type)


# ---------------------------------------------------------------- HTML
_SKIP_TAGS = frozenset({"script", "style", "head", "title", "ix:header"})
_VOID_TAGS = frozenset({"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr",
                        "param", "source", "track", "embed"})
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_BLOCK_TAGS = frozenset({"p", "div", "li", "ul", "ol", "dl", "dt", "dd", "br", "hr", "table",
                         "thead", "tbody", "tfoot", "caption", "section", "article", "aside",
                         "header", "footer", "nav", "main", "blockquote", "pre", "center", "form",
                         "fieldset", "address", "figure", "figcaption", "body", *_HEADING_TAGS})
_BOLD_STYLE = re.compile(r"font-weight\s*:\s*(?:bold|bolder|[6-9]00)\b", re.IGNORECASE)
_META_CHARSET = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE)
_XML_ENCODING = re.compile(rb"<\?xml[^>]*encoding\s*=\s*[\"']([A-Za-z0-9_.:-]+)", re.IGNORECASE)


def _is_bold(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
    # Word-generated filings bold by style rather than by <b>
    return tag in ("b", "strong") or any(k == "style" and v and _BOLD_STYLE.search(v) for k, v in attrs)


def _looks_like_heading(text: str, bold_only: bool) -> bool:
    if len(text) > MAX_HEADING_CHARS:
        return False
    letters = [c for c in text if c.isalpha()]
    return bold_only or (len(letters) >= 3 and not any(c.islower() for c in letters))


class _Blocks(HTMLParser):
    """Collects (text, kind, page) blocks in reading order. Inline tags only matter for bold.

    ``_stack`` holds the open non-void tags so that a stray or missing end tag (common in
    filings) closes what it should and nothing more. Inside a skipped element (script, head,
    ix:header) all text and structure are ignored until that element closes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, int | None]] = []
        self._stack: list[tuple[str, bool]] = []     # (tag, set in bold)
        self._open: Counter[str] = Counter()
        self._bold = 0
        self._skip_at: int | None = None             # stack index of the element being skipped
        self._buf: list[str] = []
        self._has_bold = self._has_plain = False
        self._in_row = False
        self._cells: list[str] = []

    # -- emitting
    def _end_cell(self) -> None:
        text = _collapse("".join(self._buf))
        self._buf.clear()
        if text:
            self._cells.append(text)

    def _flush(self) -> None:
        if self._in_row:
            self._end_cell()
            text, kind = " | ".join(self._cells), "table_row"
            self._cells = []
        else:
            text = _collapse("".join(self._buf))
            if any(self._open[h] for h in _HEADING_TAGS):
                kind = "heading"
            elif self._open["li"]:
                kind = "list_item"
            elif _looks_like_heading(text, self._has_bold and not self._has_plain):
                kind = "heading"
            else:
                kind = "paragraph"
        self._buf.clear()
        self._has_bold = self._has_plain = False
        if text:
            self.blocks.append((text, kind, None))

    def _cell_break(self) -> None:
        if self._in_row:
            self._end_cell()
        else:
            self._buf.append(" ")      # a stray cell outside any row: just keep the words apart

    def _boundary(self) -> None:
        if self._in_row:
            self._buf.append(" ")      # a block inside a table cell stays in the row
        else:
            self._flush()

    def finish(self) -> None:
        self.close()
        self._flush()

    # -- tag handling
    def _enter(self, tag: str) -> None:
        if tag == "tr":
            self._flush()
            self._in_row = True
        elif tag in ("td", "th"):
            self._cell_break()
        elif tag == "table":
            self._flush()
            self._in_row = False
        elif tag in _BLOCK_TAGS or tag in _SKIP_TAGS:
            self._boundary()

    def _leave(self, tag: str) -> None:
        if tag in ("tr", "table"):
            self._flush()
            self._in_row = False
        elif tag in ("td", "th"):
            self._cell_break()
        elif tag in _BLOCK_TAGS or tag in _SKIP_TAGS:
            self._boundary()

    def _pop_to(self, i: int) -> None:
        while len(self._stack) > i:
            tag, bold = self._stack.pop()
            self._open[tag] -= 1
            self._bold -= bold
        if self._skip_at is not None and i <= self._skip_at:
            self._skip_at = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body" and self._skip_at is not None:
            self._pop_to(self._skip_at)               # a <head> that was never closed
        if tag in _VOID_TAGS:
            if self._skip_at is None and tag in _BLOCK_TAGS:
                self._boundary()
            return
        skipping = self._skip_at is not None
        if not skipping:
            self._enter(tag)
        bold = not skipping and _is_bold(tag, attrs)
        self._stack.append((tag, bold))
        self._open[tag] += 1
        self._bold += bold
        if not skipping and tag in _SKIP_TAGS:
            self._skip_at = len(self._stack) - 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                break
        else:
            return                                     # end tag with no matching start
        if self._skip_at is None:
            self._leave(tag)
        self._pop_to(i)

    def handle_data(self, data: str) -> None:
        if self._skip_at is not None:
            return
        self._buf.append(data)
        if data.strip():
            if self._bold:
                self._has_bold = True
            else:
                self._has_plain = True


def _decode(raw: bytes, encoding: str | None) -> str:
    if encoding is None:
        head = raw[:16384]
        m = _META_CHARSET.search(head) or _XML_ENCODING.search(head)
        if m:
            try:
                encoding = m.group(1).decode("ascii")
                "".encode(encoding)                    # is it a codec we know?
            except (LookupError, UnicodeError):
                encoding = None
    text = raw.decode(encoding or "utf-8", errors="replace")
    return text[1:] if text.startswith("﻿") else text


def parse_html(raw: bytes, encoding: str | None = None) -> ParsedDocument:
    """Parse an SEC HTML (or inline-XBRL) filing. ``encoding`` wins over a declared charset;
    with neither, the bytes are read as UTF-8 and undecodable bytes become U+FFFD."""
    parser = _Blocks()
    parser.feed(_decode(raw, encoding))
    parser.finish()
    return _assemble(parser.blocks, raw, MEDIA_HTML)


# ---------------------------------------------------------------- text and PDF
def parse_text(raw: bytes | str) -> ParsedDocument:
    """Plain text; paragraphs are separated by blank lines. No headings are detected."""
    text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    data = raw if isinstance(raw, bytes) else raw.encode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    blocks = [(_collapse(chunk), "paragraph", None) for chunk in re.split(r"\n\s*\n", text)]
    return _assemble(blocks, data, MEDIA_TEXT)


def parse_pdf(raw: bytes) -> ParsedDocument:
    """A text PDF via the optional ``pypdf`` package: one or more paragraphs per page, with
    ``page`` set (1-based). Scanned PDFs without a text layer give no paragraphs."""
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError("parse_pdf needs the optional 'pypdf' package (pip install pypdf); "
                          "HTML and plain-text parsing do not") from exc
    blocks: list[tuple[str, str, int | None]] = []
    for number, page in enumerate(pypdf.PdfReader(io.BytesIO(raw)).pages, start=1):
        for chunk in re.split(r"\n\s*\n", page.extract_text() or ""):
            blocks.append((_collapse(chunk), "paragraph", number))
    return _assemble(blocks, raw, MEDIA_PDF)


# ---------------------------------------------------------------- matching form
# Typography folded to ASCII after NFKC; None deletes (zero-width characters and soft hyphens
# are invisible to a reader, so they must not break a match).
_FOLD = str.maketrans({
    **dict.fromkeys("‘’‚‛ʼ", "'"),
    **dict.fromkeys("“”„‟", '"'),
    **dict.fromkeys("‐‑‒–—―−﹘﹣－", "-"),
    **dict.fromkeys("​‌‍⁠﻿­", None),
})


def _fold(cluster: str) -> str:
    nfkc = unicodedata.normalize("NFKC", cluster)
    return unicodedata.normalize("NFKC", nfkc.casefold()).translate(_FOLD)


def _normalise_mapped(s: str) -> tuple[str, array, array]:
    """Normalised text plus, for each of its characters, the [start, end) span of ``s`` it came
    from. Characters are folded one cluster (base + combining marks) at a time so that NFKC
    and case folding, which can change the length, stay mapped to the right source span."""
    out: list[str] = []
    starts, ends = array("i"), array("i")
    n, i = len(s), 0
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(s[j]):
            j += 1
        chunk = s[i:j].lower() if (j == i + 1 and s[i] < "\x80") else _fold(s[i:j])
        for ch in chunk:
            if ch.isspace():
                if not out:
                    continue                           # leading whitespace
                if out[-1] == " ":
                    ends[-1] = j                       # extend the collapsed run
                    continue
                ch = " "
            out.append(ch)
            starts.append(i)
            ends.append(j)
        i = j
    if out and out[-1] == " ":
        out.pop()
        starts.pop()
        ends.pop()
    return "".join(out), starts, ends


def normalise(s: str) -> str:
    """The form quotes are matched in: NFKC, curly quotes and dashes folded to ASCII,
    case folded, whitespace collapsed and stripped."""
    return _normalise_mapped(s)[0]


class _NormIndex:
    """A document's normalised text with the map back to its original text, built once."""

    def __init__(self, doc: ParsedDocument) -> None:
        self.source = doc.text
        self.text, self.starts, self.ends = _normalise_mapped(doc.text)
        self.para_starts = [p.char_start for p in doc.paragraphs]

    def aligned(self, a: int, b: int) -> bool:
        """True if [a, b) neither starts nor ends inside one original character's expansion
        (the "ss" that "ß" folds to, say)."""
        return ((a == 0 or self.starts[a] != self.starts[a - 1])
                and (b == len(self.text) or self.starts[b] != self.starts[b - 1]))


def _whole_tokens(n: str, a: int, b: int) -> bool:
    """A match must not begin or end in the middle of a word or number."""
    first, last = n[a], n[b - 1]
    if a > 0:
        prev = n[a - 1]
        if first.isalnum() and (prev.isalnum() or (prev in ".," and a > 1 and n[a - 2].isdigit()
                                                   and first.isdigit())):
            return False
        if first in ".," and prev.isdigit() and a + 1 < b and n[a + 1].isdigit():
            return False                               # ".5%" inside "0.5%"
    if b < len(n):
        nxt = n[b]
        if last.isalnum() and (nxt.isalnum() or (nxt in ".," and b + 1 < len(n)
                                                 and n[b + 1].isdigit() and last.isdigit())):
            return False
    return True


def _occurrences(doc: ParsedDocument, quote: str) -> Iterator[tuple[int, int]]:
    """Non-overlapping (start, end) spans of the quote in the document's normalised text."""
    q = normalise(quote)
    if not q:
        return
    ix = doc._index()
    pos = ix.text.find(q)
    while pos != -1:
        end = pos + len(q)
        if ix.aligned(pos, end) and _whole_tokens(ix.text, pos, end):
            yield pos, end
            pos = ix.text.find(q, end)
        else:
            pos = ix.text.find(q, pos + 1)


def locate_quote(doc: ParsedDocument, quote: str) -> Locator | None:
    """Where the quote first occurs, or None. Ignores case, whitespace and quote/dash
    typography; may span adjacent paragraphs; never matches fuzzily."""
    hit = next(_occurrences(doc, quote), None)
    if hit is None:
        return None
    ix = doc._index()
    a, b = ix.starts[hit[0]], ix.ends[hit[1] - 1]
    first = doc.paragraphs[bisect.bisect_right(ix.para_starts, a) - 1]
    last = doc.paragraphs[bisect.bisect_right(ix.para_starts, b - 1) - 1]
    return Locator(first.index, last.index, a, b, first.page, first.heading, doc.text[a:b])


def count_quote(doc: ParsedDocument, quote: str) -> int:
    """How many times the quote occurs (same matching as ``locate_quote``)."""
    return sum(1 for _ in _occurrences(doc, quote))


# ---------------------------------------------------------------- numbers
_SMALL = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
          "fifteen sixteen seventeen eighteen nineteen twenty").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_WORDS = {**{w: i for i, w in enumerate(_SMALL)}, **_TENS, "hundred": 100}
_DENOMINATOR = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4,
                "fourth": 4, "fourths": 4, "fifth": 5, "fifths": 5, "sixth": 6, "sixths": 6,
                "seventh": 7, "sevenths": 7, "eighth": 8, "eighths": 8, "ninth": 9, "ninths": 9,
                "tenth": 10, "tenths": 10}


def _alt(words) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


_DIGIT_UNITS = _alt(_SMALL[1:10])
# Alternatives are tried in order at each position, and the digit groups are atomic so that a
# rejected token ("12b", "1,250.5x") is skipped whole instead of yielding a shorter number.
_NUMBER = re.compile(rf"""
    (?P<mixed>(?<![\w.])(?>(?P<mw>\d+))[\s-](?>(?P<mn>\d{{1,2}}))/(?>(?P<md>\d{{1,2}}))(?![\w/]))
  | (?P<frac>(?<![\w./])(?>(?P<fnum>\d{{1,3}}))/(?>(?P<fden>\d{{1,3}}))(?![\w/]))
  | (?P<dig>(?<!\w)(?>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?!\w))
  | (?P<fracword>\b(?P<wn>{_DIGIT_UNITS})[\s-]+(?P<wd>{_alt(_DENOMINATOR)})\b)
  | (?P<hundred>\b(?:(?P<hm>{_DIGIT_UNITS})[\s-]+)?hundred\b)
  | (?P<tens>\b(?P<tw>{_alt(_TENS)})[\s-]+(?P<tu>{_DIGIT_UNITS})\b)
  | (?P<word>\b(?:{_alt(_WORDS)})\b)
""", re.IGNORECASE | re.VERBOSE)
_PERCENT = re.compile(r"\s*\)?\s*(?:%|percent(?:age)?\b|per\s?cent\b)", re.IGNORECASE)
_ONE_AFTER = frozenset({"any", "each", "every", "no", "the", "this", "that", "which", "per"})
_ONE_BEFORE = frozenset({"of", "or", "another"})


@dataclass(frozen=True)
class _Found:
    value: float
    percent: bool = False       # written as a percentage: 80% also reads as 0.8
    fraction: bool = False      # written as a fraction below 1: one-quarter also reads as 25


def _one_is_number(text: str, start: int, end: int) -> bool:
    """"one" is also a pronoun ("any one issuer", "one of the"); read it as a number only where
    it clearly is one. Doubt costs a flagged threshold, never a wrongly accepted one."""
    if text[end:end + 1] == "-" or (start > 0 and text[start - 1] == "-"):
        return False
    prev = re.search(r"([A-Za-z]+)\W*$", text[max(0, start - 24):start])
    nxt = re.match(r"\s+([A-Za-z]+)", text[end:end + 24])
    return not ((prev and prev.group(1).lower() in _ONE_AFTER)
                or (nxt and nxt.group(1).lower() in _ONE_BEFORE))


def _scan(text: str) -> list[_Found]:
    found: list[_Found] = []

    def add(value: float, end: int, fraction: bool = False) -> None:
        found.append(_Found(value, bool(_PERCENT.match(text, end)), fraction))

    for m in _NUMBER.finditer(text):
        start, end = m.span()
        if m["mixed"] is not None:
            whole, n, d = int(m["mw"]), int(m["mn"]), int(m["md"])
            if 0 < n < d <= 20:
                add(whole + n / d, end)                # "33 1/3%" is 33.33...
            else:
                found.extend([_Found(float(whole)), _Found(float(n))])
                add(float(d), end)
        elif m["frac"] is not None:
            n, d = int(m["fnum"]), int(m["fden"])
            if 0 < n < d <= 20:
                add(n / d, end, fraction=True)
            else:
                found.append(_Found(float(n)))
                add(float(d), end)
        elif m["dig"] is not None:
            # "12b-1", "35d-1", "N-14" and "10-K" are identifiers, not quantities
            if re.search(r"[A-Za-z]-$", text[max(0, start - 2):start]) or re.match(r"-[A-Z]", text[end:end + 2]):
                continue
            add(float(m["dig"].replace(",", "")), end)
        elif m["fracword"] is not None:
            n, d = m["wn"].lower(), _DENOMINATOR[m["wd"].lower()]
            if n == "one":
                add(1 / d, end, fraction=True)
            elif n == "three" and d == 4:
                add(0.75, end, fraction=True)
            # any other ("two-thirds") is skipped whole, not misread as 2 and thirds
        elif m["hundred"] is not None:
            add(100.0 * (_WORDS[m["hm"].lower()] if m["hm"] else 1), end)
        elif m["tens"] is not None:
            add(float(_TENS[m["tw"].lower()] + _WORDS[m["tu"].lower()]), end)
        else:
            word = m["word"].lower()
            if word != "one" or _one_is_number(text, start, end):
                add(float(_WORDS[word]), end)
    return found


def numbers_in(text: str) -> list[float]:
    """The numbers a reader sees in the text, in order: digits ("1,250.5"), percentages
    ("80%", "80 percent": 80.0), spelled-out numbers ("eighty", "twenty-five", "one hundred"),
    and fractions ("one-third", "1/3", "three-quarters", "33 1/3": the fraction's value).
    Not returned: digits inside identifiers ("35d-1", "12b-1", "S000000856"), the parts of
    fractions this does not read ("two-thirds"), and "one" used as a pronoun ("any one issuer")."""
    return [f.value for f in _scan(text)]


def number_appears(value: float, text: str, rel_tol: float = 1e-9) -> bool:
    """True if ``value`` matches a number in the text directly, or as percent <-> fraction:
    0.8 and 80 both match "80%"; 0.25 and 25 both match "25 percent" and "one-quarter".
    A bare "80" (no percent sign or word) matches 80 only."""
    for f in _scan(text):
        candidates = [f.value]
        if f.percent:
            candidates.append(f.value / 100)
        if f.fraction:
            candidates.append(f.value * 100)
        if any(math.isclose(value, c, rel_tol=rel_tol, abs_tol=0.0) for c in candidates):
            return True
    return False
