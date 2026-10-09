import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fairbench.ingest.documents import (Locator, count_quote, locate_quote, normalise,
                                        number_appears, numbers_in, parse_html, parse_pdf,
                                        parse_text)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prospectus_synthetic.htm"

STRATEGY = "Principal Investment Strategies"
RESTRICTIONS = "FUNDAMENTAL INVESTMENT RESTRICTIONS"
SENTENCES = {
    "invest_80": ("Under normal circumstances, the Fund invests at least 80% of its net assets in "
                  "equity securities of large-capitalization companies.", STRATEGY),
    "tobacco": ("The Fund will not invest in companies that derive 10% or more of their revenue "
                "from the manufacture of tobacco products.", STRATEGY),
    "carbon": ("The Fund seeks to maintain a portfolio carbon intensity below that of its "
               "benchmark.", STRATEGY),
    "holdings": ("The Fund normally holds between 35 and 45 securities.", STRATEGY),
    "industry": ("The Fund may not invest more than 25% of its total assets in any one industry.",
                 RESTRICTIONS),
    "issuer": ("With respect to 75% of its total assets, the Fund may not invest more than 5% of "
               "its total assets in the securities of any one issuer.", RESTRICTIONS),
    "esg": ("The Adviser considers environmental, social and governance factors as part of its "
            "investment process.", STRATEGY),
    "borrow": ("The Fund may not borrow money in excess of 33 1/3% of the value of its total "
               "assets.", RESTRICTIONS),
}
# One sentence set in two <div>s, with curly quotes, curly apostrophes and em dashes.
SPANNING = ("The Adviser treats an issuer as a “controversial business” if the Adviser’s "
            "research — which the Adviser may update at any time — shows that the issuer "
            "fails the Fund’s screens, and the Fund will then sell the issuer’s securities "
            "within a reasonable period.")


@pytest.fixture(scope="module")
def raw() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def doc(raw):
    return parse_html(raw)


# ---------------------------------------------------------------- HTML structure
def test_source_hash_and_media_type(doc, raw):
    assert doc.sha256_of_source == hashlib.sha256(raw).hexdigest()
    assert doc.media_type == "text/html"


def test_script_style_title_and_ix_header_are_dropped_but_ix_text_is_kept(doc):
    for marker in ("SCRIPT_MARKER", "STYLE_MARKER", "TITLE_MARKER", "IX_HEADER_MARKER",
                   "0000000000", "document.write"):
        assert marker not in doc.text
    # ix:nonNumeric as a block wrapper and as an inline span in the middle of a sentence
    assert SENTENCES["invest_80"][0] in doc.text
    assert SENTENCES["carbon"][0] in doc.text
    assert "<" not in doc.text and "&" not in doc.text


def test_entities_and_spaces_are_normalised(doc):
    assert "80% of its net assets" in doc.text            # &nbsp; became a plain space
    assert " " not in doc.text and "  " not in doc.text
    assert "“controversial business”" in doc.text   # &#8220; &#8221; decoded
    assert "Ticker symbol: SYNEX (fictional)" in doc.text


def test_paragraph_offsets_index_the_full_text(doc):
    assert doc.text == "\n".join(p.text for p in doc.paragraphs)
    assert [p.index for p in doc.paragraphs] == list(range(len(doc.paragraphs)))
    for p in doc.paragraphs:
        assert p.text and p.text == p.text.strip()
        assert doc.text[p.char_start:p.char_end] == p.text
        assert p.page is None
    for a, b in zip(doc.paragraphs, doc.paragraphs[1:]):
        assert b.char_start == a.char_end + 1               # one "\n" between paragraphs


def test_headings_are_detected_and_attached_to_following_paragraphs(doc):
    kinds = {p.text: p for p in doc.paragraphs if p.kind == "heading"}
    assert set(kinds) == {"PROSPECTUS", STRATEGY, "Annual Fund Operating Expenses",
                          RESTRICTIONS, "Portfolio Management"}
    by_text = {p.text: p for p in doc.paragraphs}
    for key, (sentence, heading) in SENTENCES.items():
        assert by_text[sentence].heading == heading, key
        assert by_text[sentence].kind == "paragraph", key
    assert by_text["Ticker symbol: SYNEX (fictional)"].heading == "PROSPECTUS"
    # a heading belongs to the section it opens
    assert all(p.heading == p.text for p in kinds.values())
    assert doc.paragraphs[0].kind == "heading" and doc.paragraphs[0].heading == "PROSPECTUS"
    assert doc.paragraphs[-1].heading == "Portfolio Management"


def test_bold_lead_in_does_not_make_a_heading(doc):
    note = next(p for p in doc.paragraphs if p.text.startswith("Note:"))
    assert note.kind == "paragraph" and note.heading == STRATEGY


def test_section_returns_the_strategy_paragraphs(doc):
    strategy = doc.section("principal investment strateg")
    texts = [p.text for p in strategy]
    for key in ("invest_80", "tobacco", "carbon", "holdings", "esg"):
        assert SENTENCES[key][0] in texts
    assert SENTENCES["industry"][0] not in texts
    assert texts[0] == STRATEGY                               # the title line opens its section...
    assert "Annual Fund Operating Expenses" not in texts      # ...and the next title does not leak in
    assert all(p.heading == STRATEGY for p in strategy)
    restrictions = doc.section("fundamental (investment )?(restrictions|policies)")
    assert [p.text for p in restrictions] == [RESTRICTIONS] + [SENTENCES[k][0] for k in ("industry", "issuer", "borrow")]
    assert doc.section("PRINCIPAL INVESTMENT STRATEG") == strategy      # case-insensitive
    assert doc.section("no such heading") == []


def test_list_items_and_table_rows(doc):
    items = [p for p in doc.paragraphs if p.kind == "list_item"]
    assert [p.text for p in items] == [
        "American depositary receipts of large-capitalization companies;",
        "exchange-traded funds that hold equity securities; and",
        "cash equivalents held for liquidity.",
    ]
    assert all(p.heading == STRATEGY for p in items)
    rows = [p for p in doc.paragraphs if p.kind == "table_row"]
    assert [p.text for p in rows] == [
        "Fee item | Percent of average net assets",           # bold cells do not make a heading
        "Management fee | 0.50%",                             # the empty &nbsp; cell is dropped
        "Distribution and service (Rule 12b-1) fees | 0.25%",
        "Total annual fund operating expenses | 0.75%",
    ]
    assert all(p.heading == "Annual Fund Operating Expenses" for p in rows)


def test_br_splits_paragraphs(doc):
    texts = [p.text for p in doc.paragraphs]
    assert "Synthetic Example Equity Fund (not a real fund)" in texts
    assert "Ticker symbol: SYNEX (fictional)" in texts


# ---------------------------------------------------------------- HTML edge cases
def test_heading_heuristics_on_small_inputs():
    html = (b"<div><b>Bold Only Heading</b></div><div>ALL CAPS HEADING</div><div>AI</div>"
            b"<div>Plain sentence.</div><div><span style='font-weight:700'>Styled Bold</span></div>"
            b"<div><b>" + b"x" * 130 + b"</b></div>")
    d = parse_html(html)
    assert [(p.text[:20], p.kind) for p in d.paragraphs] == [
        ("Bold Only Heading", "heading"), ("ALL CAPS HEADING", "heading"),
        ("AI", "paragraph"),                                  # fewer than 3 letters
        ("Plain sentence.", "paragraph"), ("Styled Bold", "heading"),
        ("x" * 20, "paragraph"),                              # bold but too long
    ]
    assert d.paragraphs[3].heading == "ALL CAPS HEADING"


def test_unclosed_tags_and_nested_blocks_in_cells():
    html = (b"<html><head><title>gone</title><body><p>one<p>two<ul><li>a<li>b</ul><p>three"
            b"<table><tr><td><div>cell</div><p>text</p><td>second<tr><td>x</table>tail")
    d = parse_html(html)
    assert [(p.text, p.kind) for p in d.paragraphs] == [
        ("one", "paragraph"), ("two", "paragraph"), ("a", "list_item"), ("b", "list_item"),
        ("three", "paragraph"), ("cell text | second", "table_row"), ("x", "table_row"),
        ("tail", "paragraph")]


def test_encoding_declared_explicit_and_default():
    body = "<p>Café — 80% net</p>"
    latin = f'<meta charset="iso-8859-1">{body.replace(chr(0x2014), "-")}'.encode("latin-1")
    assert parse_html(latin).paragraphs[0].text == "Café - 80% net"
    assert parse_html(body.encode("cp1252", errors="replace"), encoding="cp1252").paragraphs[0].text.startswith("Café")
    assert parse_html(body.encode("utf-8")).paragraphs[0].text == "Café — 80% net"
    assert parse_html(b"<p>bad \xff byte</p>").paragraphs[0].text == "bad � byte"
    assert parse_html(b"<meta charset=nonsense-9><p>ok</p>").paragraphs[0].text == "ok"
    assert parse_html(b"\xef\xbb\xbf<p>bom</p>").text == "bom"
    assert parse_html(b"<script>x</script>").paragraphs == []


# ---------------------------------------------------------------- normalise
def test_normalise_folds_typography_case_and_space():
    assert normalise("  The  Adviser’s “Screen” — and – more \n done ") == \
        'the adviser\'s "screen" - and - more done'
    assert normalise("‘a’ ‐ − b") == "'a' - - b"
    assert normalise("Straße") == "strasse"
    assert normalise("ﬁnal…") == "final..."
    assert normalise("in​vest­") == "invest"
    assert normalise("é") == normalise("é") == "é"
    assert normalise("") == normalise(" \n\t") == ""


# ---------------------------------------------------------------- locate_quote
@pytest.mark.parametrize("key", sorted(SENTENCES))
def test_locate_each_sentence_exactly(doc, key):
    sentence, heading = SENTENCES[key]
    loc = locate_quote(doc, sentence)
    assert isinstance(loc, Locator)
    para = doc.paragraphs[loc.paragraph_index]
    assert para.text == sentence
    assert loc.paragraph_end_index == loc.paragraph_index
    assert (loc.char_start, loc.char_end) == (para.char_start, para.char_end)
    assert loc.text == sentence == doc.text[loc.char_start:loc.char_end]
    assert loc.heading == heading and loc.page is None
    assert loc.as_string() == f'para {para.index} under "{heading}"'
    assert count_quote(doc, sentence) == 1


def test_locate_a_fragment_inside_a_paragraph(doc):
    loc = locate_quote(doc, "may not invest more than 5% of its total assets in the securities")
    assert loc is not None
    assert loc.text == "may not invest more than 5% of its total assets in the securities"
    assert doc.paragraphs[loc.paragraph_index].text == SENTENCES["issuer"][0]
    assert loc.char_start > doc.paragraphs[loc.paragraph_index].char_start


def test_locate_tolerates_whitespace_case_and_quote_typography(doc):
    sentence = SENTENCES["industry"][0]
    exact = locate_quote(doc, sentence)
    for variant in (sentence.upper(), sentence.lower(), "  " + sentence.replace(" ", "  \n\t ") + "\n",
                    sentence.replace(" ", " ")):
        assert locate_quote(doc, variant) == exact
    # straight quotes and dashes in the quote, curly ones in the document, and the reverse
    straight = ("The Adviser treats an issuer as a \"controversial business\" if the Adviser's "
                "research - which the Adviser may update at any time - shows that")
    loc = locate_quote(doc, straight)
    assert loc is not None and "“controversial business”" in loc.text
    assert loc.text == doc.text[loc.char_start:loc.char_end]
    assert locate_quote(doc, SPANNING.replace("“", '"').replace("”", '"')
                        .replace("’", "'").replace("—", "--")) is None   # "--" is not a dash fold
    assert locate_quote(doc, SPANNING.replace("—", "-").replace("“", '"')
                        .replace("”", '"').replace("’", "'")) is not None
    assert locate_quote(doc, "33 1/3% of the value") is not None


def test_locate_spans_two_paragraphs(doc):
    loc = locate_quote(doc, SPANNING)
    assert loc is not None
    assert loc.paragraph_end_index == loc.paragraph_index + 1
    assert loc.text == doc.text[loc.char_start:loc.char_end]
    assert "shows that\nthe issuer fails" in loc.text           # the document's own text, with its "\n"
    first, last = doc.paragraphs[loc.paragraph_index], doc.paragraphs[loc.paragraph_end_index]
    assert loc.char_start == first.char_start and loc.char_end == last.char_end
    assert loc.as_string() == f'paras {first.index}-{last.index} under "{STRATEGY}"'
    # a quote that starts in one paragraph and stops inside the next
    part = locate_quote(doc, "shows that the issuer fails the Fund’s screens")
    assert part is not None and part.paragraph_end_index == part.paragraph_index + 1


@pytest.mark.parametrize("quote", [
    # paraphrase of a real sentence
    "The Fund puts at least 80% of net assets into large-cap equities.",
    "The Fund invests at least eighty percent of its net assets in equity securities.",
    # one changed number
    SENTENCES["invest_80"][0].replace("80%", "90%"),
    SENTENCES["industry"][0].replace("25%", "20%"),
    SENTENCES["tobacco"][0].replace("10%", "5%"),
    SENTENCES["holdings"][0].replace("45", "46"),
    # a changed word, an extra word, a dropped word
    SENTENCES["esg"][0].replace("considers", "ignores"),
    SENTENCES["carbon"][0].replace("below", "well below"),
    SENTENCES["issuer"][0].replace("the Fund may not", "may not"),
    # starts or ends inside a number or a word
    "5% of its total assets in any one industry",             # the document says 25%
    "0% of its net assets in equity securities",              # the document says 80%
    "The Fund normally holds between 35 and 4",               # the document says 45
    "he Fund may not invest more than 25%",
    # not in the document at all, and empty quotes
    "The Fund may use leverage.", "", "   ",
])
def test_locate_returns_none_for_what_the_document_does_not_say(doc, quote):
    assert locate_quote(doc, quote) is None
    assert count_quote(doc, quote) == 0


def test_a_decimal_point_is_not_a_boundary():
    d = parse_text("Fees are 2.5% of assets. The cap is 1,250 shares. Up to 33.5% is allowed.")
    assert locate_quote(d, "5% of assets") is None
    assert locate_quote(d, ".5% of assets") is None
    assert locate_quote(d, "2.5% of assets") is not None
    assert locate_quote(d, "250 shares") is None
    assert locate_quote(d, "cap is 1") is None
    assert locate_quote(d, "cap is 1,250") is not None
    assert locate_quote(d, "33.5%") is not None and locate_quote(d, "33.5") is not None
    assert locate_quote(d, "Up to 33") is None


def test_count_quote_and_first_occurrence():
    d = parse_text("The Fund may not borrow.\n\nOther text.\n\nthe fund MAY NOT borrow.\n\n"
                   "The Fund may not borrow money. The Fund may not borrow.")
    assert count_quote(d, "The Fund may not borrow") == 4
    assert count_quote(d, "the fund may not borrow.") == 3          # the 4th has no full stop
    first = locate_quote(d, "The Fund may not borrow")
    assert (first.paragraph_index, first.char_start) == (0, 0)
    assert first.text == "The Fund may not borrow"
    third = locate_quote(d, "the fund may not borrow.")
    assert third.paragraph_index == 0
    assert count_quote(d, "nothing like this") == 0
    # overlapping candidates are counted once each
    assert count_quote(parse_text("aaaa aaaa aaaa"), "aaaa") == 3


def test_offsets_refer_to_the_original_text_when_normalisation_changes_length():
    d = parse_text("Wait…  the ﬁnal   rule applies; the Straße rule too.")
    assert d.text == "Wait… the ﬁnal rule applies; the Straße rule too."
    loc = locate_quote(d, "THE FINAL RULE APPLIES")
    assert loc.text == "the ﬁnal rule applies" == d.text[loc.char_start:loc.char_end]
    loc = locate_quote(d, "wait... the")
    assert loc.text == "Wait… the"
    assert locate_quote(d, "the strasse rule").text == "the Straße rule"
    # a match may not start or end inside the characters one source character expands to
    assert locate_quote(d, "stras") is None and locate_quote(d, "asse rule") is None
    assert locate_quote(d, "wait.. ") is None and locate_quote(d, "ait...") is None
    assert locate_quote(d, "fi") is None and locate_quote(d, "nal rule") is None
    d2 = parse_text("in​vest­ment policy")
    loc = locate_quote(d2, "investment policy")
    assert loc.text == "in​vest­ment policy" and loc.char_start == 0


def test_locator_as_string_formats():
    base = dict(char_start=0, char_end=1, text="x")
    assert Locator(14, 14, page=None, heading=None, **base).as_string() == "para 14"
    assert Locator(14, 15, page=None, heading=None, **base).as_string() == "paras 14-15"
    assert Locator(14, 14, page=3, heading=None, **base).as_string() == "page 3, para 14"
    assert Locator(14, 15, page=3, heading="Risks", **base).as_string() == 'page 3, paras 14-15 under "Risks"'


def test_the_normalised_index_is_built_once_and_rebuilt_if_the_text_changes():
    d = parse_text("Alpha beta.\n\nGamma delta.")
    assert locate_quote(d, "beta") is not None
    index = d._index()
    assert locate_quote(d, "gamma") is not None and d._index() is index
    d.text = d.text.replace("Gamma", "Omega")
    assert locate_quote(d, "gamma") is None and d._index() is not index


# ---------------------------------------------------------------- numbers
@pytest.mark.parametrize("text, expected", [
    ("80%", [80.0]),
    ("80 percent", [80.0]),
    ("eighty percent", [80.0]),
    ("Eighty Percent of net assets", [80.0]),
    ("1,250.5", [1250.5]),
    ("$1,250,000 and 1,250.5%", [1250000.0, 1250.5]),
    ("0.25% and .75%", [0.25, 0.75]),
    ("between 35 and 45 securities", [35.0, 45.0]),
    ("35-45 securities", [35.0, 45.0]),
    ("a 5-year average and a 30-day yield", [5.0, 30.0]),
    ("at least five but no more than twenty holdings", [5.0, 20.0]),
    ("zero, ten, thirteen, ninety", [0.0, 10.0, 13.0, 90.0]),
    ("twenty-five percent", [25.0]),
    ("forty five percent", [45.0]),
    ("one hundred percent or two hundred", [100.0, 200.0]),
    ("one-third", [1 / 3]),
    ("one third of assets", [1 / 3]),
    ("1/3", [1 / 3]),
    ("3/4 of the portfolio", [0.75]),
    ("one-quarter", [0.25]),
    ("one fourth", [0.25]),
    ("one half", [0.5]),
    ("one-half of one percent", [0.5, 1.0]),
    ("three-quarters", [0.75]),
    ("33 1/3% of total assets", [33 + 1 / 3]),
    ("33-1/3%", [33 + 1 / 3]),
    ("2024 12/31 report", [2024.0, 12.0, 31.0]),          # not a fraction (12/31 is a date)
    ("section 4.5 and 12/31/2024", [4.5, 12.0, 31.0, 2024.0]),
    # not numbers: identifiers, and fractions this does not read
    ("35d-1", []),
    ("Rule 35d-1 and Rule 12b-1", []),
    ("12b-1", []),
    ("S000000856", []),
    ("Form N-14, Form 10-K and 485BPOS and Rule 144A", []),
    ("the 10th day", []),
    ("1,250.5x", []),
    ("two-thirds", []),
    ("two thirds of the shares", []),
    ("any one issuer, each one, no one", []),
    ("one of the following, one or more, a one-time fee", []),
    ("Rule 12b-1 fees of 0.25% annually", [0.25]),
    ("", []),
    ("no numbers here", []),
])
def test_numbers_in(text, expected):
    got = numbers_in(text)
    assert got == pytest.approx(expected, rel=1e-12)
    assert len(got) == len(expected)


def test_numbers_in_reads_the_fixture_sentences(doc):
    for key, expected in {"invest_80": [80.0], "tobacco": [10.0], "carbon": [], "holdings": [35.0, 45.0],
                          "industry": [25.0], "issuer": [75.0, 5.0], "esg": [],
                          "borrow": [33 + 1 / 3]}.items():
        assert numbers_in(SENTENCES[key][0]) == pytest.approx(expected), key
    policy = doc.paragraphs[-1].text
    assert "35d-1" in policy and numbers_in(policy) == [0.25]


def test_numbers_in_returns_floats_in_order_of_appearance():
    got = numbers_in("first 7, then fifty percent, then 1/4, then 9.5")
    assert got == [7.0, 50.0, 0.25, 9.5]
    assert all(isinstance(x, float) for x in got)


def test_number_appears_directly_and_as_percent_or_fraction():
    # percent <-> fraction
    for text in ("80%", "80 percent", "eighty percent", "eighty per cent", "80 % of assets"):
        assert number_appears(80, text) and number_appears(80.0, text) and number_appears(0.8, text)
    assert number_appears(0.25, "25 percent") and number_appears(25, "25 percent")
    assert number_appears(0.25, "one-quarter") and number_appears(25, "one-quarter")
    assert number_appears(0.75, "three-quarters") and number_appears(1 / 3, "one-third")
    assert number_appears(1 / 3, "1/3") and number_appears(0.5, "one half")
    assert number_appears(0.5, "0.5%") and number_appears(0.005, "0.5%")
    assert number_appears(0.0025, "0.25 percent")
    assert number_appears(33 + 1 / 3, "33 1/3%") and number_appears((33 + 1 / 3) / 100, "33 1/3%")
    assert number_appears(100, "one hundred percent") and number_appears(1.0, "100%")
    # plain numbers
    assert number_appears(35, "between 35 and 45 securities") and number_appears(45, "between 35 and 45")
    assert number_appears(1250.5, "1,250.5") and number_appears(5, "twenty-five percent") is False
    assert number_appears(25, "twenty-five percent")
    assert number_appears(0, "zero holdings")


def test_number_appears_is_false_for_values_not_in_the_text(doc):
    sentence = SENTENCES["invest_80"][0]
    assert not number_appears(90, sentence) and not number_appears(0.9, sentence)
    assert not number_appears(8, sentence) and not number_appears(800, sentence)
    assert not number_appears(0.08, sentence)
    assert not number_appears(0.8, "80 securities")            # a bare 80 is not a percentage
    assert not number_appears(0.8, "between 80 and 90")
    assert not number_appears(80, "0.8 times the benchmark")
    assert not number_appears(35, "Rule 35d-1")
    assert not number_appears(1, "Rule 12b-1") and not number_appears(12, "Rule 12b-1")
    assert not number_appears(2, "two-thirds") and not number_appears(3, "two-thirds")
    assert not number_appears(2 / 3, "two-thirds")
    assert not number_appears(1, "any one issuer")
    assert not number_appears(25, "a quarter") and not number_appears(80, "")
    assert not number_appears(0.2, "one-quarter") and not number_appears(250, "one-quarter")
    assert not number_appears(float("nan"), "nan 5")
    assert number_appears(0.33, "one-third") is False
    assert number_appears(0.33, "one-third", rel_tol=0.02)
    assert not number_appears(80.0001, "80%") and number_appears(80.0001, "80%", rel_tol=1e-5)
    # a threshold read off the quoted sentence appears in it; one read off the wrong sentence does not
    loc = locate_quote(doc, SENTENCES["tobacco"][0])
    assert number_appears(0.10, loc.text) and not number_appears(0.25, loc.text)


# ---------------------------------------------------------------- text and PDF
def test_parse_text_splits_on_blank_lines():
    raw = "First  paragraph\nstill first.\r\n\r\n\n  \nSecond paragraph.\n\n\n\nThird."
    d = parse_text(raw)
    assert [p.text for p in d.paragraphs] == ["First paragraph still first.", "Second paragraph.", "Third."]
    assert d.media_type == "text/plain"
    assert d.text == "\n".join(p.text for p in d.paragraphs)
    assert all(d.text[p.char_start:p.char_end] == p.text for p in d.paragraphs)
    assert all(p.heading is None and p.kind == "paragraph" and p.page is None for p in d.paragraphs)
    assert d.sha256_of_source == hashlib.sha256(raw.encode()).hexdigest()
    as_bytes = parse_text(raw.encode("utf-8"))
    assert as_bytes.text == d.text and as_bytes.sha256_of_source == d.sha256_of_source
    assert parse_text("").paragraphs == [] and parse_text(" \n\n ").paragraphs == []
    assert locate_quote(d, "second paragraph. third").paragraph_end_index == 2


def test_parse_pdf_without_pypdf_raises_a_helpful_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)             # makes `import pypdf` fail
    with pytest.raises(ImportError, match="pypdf"):
        parse_pdf(b"%PDF-1.4")
    with pytest.raises(ImportError, match="pip install pypdf"):
        parse_pdf(b"%PDF-1.4")


def test_parse_pdf_paragraphs_carry_page_numbers(monkeypatch):
    pages = ["Fund summary\nwraps here.\n\nThe Fund may not\nborrow money.", "", "Page three text."]

    class FakeReader:
        def __init__(self, stream):
            assert stream.read() == b"%PDF fake"
            self.pages = [SimpleNamespace(extract_text=lambda t=t: t) for t in pages]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))
    d = parse_pdf(b"%PDF fake")
    assert d.media_type == "application/pdf"
    assert [(p.page, p.text) for p in d.paragraphs] == [
        (1, "Fund summary wraps here."), (1, "The Fund may not borrow money."), (3, "Page three text.")]
    assert d.sha256_of_source == hashlib.sha256(b"%PDF fake").hexdigest()
    assert all(d.text[p.char_start:p.char_end] == p.text for p in d.paragraphs)
    loc = locate_quote(d, "may not borrow money")
    assert loc.page == 1 and loc.as_string() == "page 1, para 1"
    assert locate_quote(d, "page three").as_string() == "page 3, para 2"
