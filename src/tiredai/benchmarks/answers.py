"""Reading an answer deterministically: which catalog products it names, and whether the facts it states are true.

A product counts as named when its line ("Michelin Defender 2", see catalog.product_line) or its SKU
appears in the answer. When several products the agent saw share that line, the size, the rest of the
name and the price written next to it narrow them down; a mention that still fits several keeps all of
them as candidates.

Facts are checked per line of the answer. A line that names exactly one product owns it, and so do the
lines after it in the same paragraph until another product is named (a product card with its specs on
separate lines). A table row is only about the product it names. Checked facts:
- prices: must be the owning product's price (or 2 or 4 tires of it), or some seen product's price on a
  line that doesn't name a product, or a number the shopper wrote (a budget). A price without cents may
  be rounded; one with cents may also be the difference between two shown prices (a saving). A price
  after "under", "up to" and the like is a bound, not a fact, and isn't checked.
- SKUs: must be among the products the agent saw.
- tread depth (n/32), mileage warranty, UTQG and recommendation level (n/5): checked against the owning
  product only, since without one they can be general knowledge.
"""

import math
import re
from dataclasses import dataclass

from tiredai.search import parse_size

PRICE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d{1,2})?")
SKU = re.compile(r"\b[A-Z]{1,2}\d{5,8}-\d{2}\b")
TREAD = re.compile(r"\b(\d{1,2})\s?/\s?32")
WARRANTY = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{2,3}(?:\.\d)?\s?[kK])\s?(?:-|\s)?miles?\b")
UTQG = re.compile(r"\b(\d{3}) ?([ABC]{1,2}) ?([ABC])\b")
RECOMMENDATION = re.compile(r"\b([1-5])\s?/\s?5\b")
SERVICE = re.compile(r"\b\d{2,3}(?:/\d{2,3})?[A-Z]\d?\b")  # load index and speed rating, e.g. 92V or 121/118Q
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
UNAVAILABLE = ("out of stock", "not in stock", "unavailable", "not available", "sold out", "available: false")
NO_STOCK_CELL = re.compile(r"\b(no|out|false|unavailable)\b|❌|✗|✖")  # a table's in-stock column saying no
STOCK_HEADER = re.compile(r"avail|stock")
SETS = (1, 2, 4)  # a price can be quoted per tire, per pair or per set of four
# A price after these words is a bound ("the three options under $90"), not a product's price.
BOUND = re.compile(r"\b(under|below|less than|up to|within|over|above|more than|at most|at least|max|min)\s*$")


def plain(text: str) -> str:
    """Lowercase, straight quotes, no Markdown emphasis, single spaces: for phrase checks."""
    text = text.replace("’", "'").replace("‘", "'").lower()
    return " ".join(re.sub(r"[*_`]", "", text).split())


def has_phrase(text: str, phrase: str) -> bool:
    """`phrase` in `text` as whole words (a plural s is fine), ignoring case and Markdown: 'tire' isn't in 'TiredAI'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(plain(phrase))}(?:e?s)?(?![a-z0-9])", plain(text)) is not None


def words(text: str) -> str:
    """Lowercase words and numbers only, for matching names; slashes and decimal points stay ('h/t', '5.20-13')."""
    text = re.sub(r"[^a-z0-9/.]+", " ", text.lower())
    return " ".join(re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text).split())


def contains_words(text: str, phrase: str) -> bool:
    return f" {words(phrase)} " in f" {words(text)} "


@dataclass(frozen=True)
class Mention:
    line: int  # index in the answer's lines
    skus: tuple[str, ...]  # the products this mention can be
    unavailable: bool  # the line says it is out of stock, so it isn't a recommendation


@dataclass(frozen=True)
class Fact:
    kind: str
    text: str
    supported: bool


def paragraphs(answer: str) -> list[list[str]]:
    blocks = re.split(r"\n\s*\n", answer.strip())
    return [[line for line in block.splitlines() if line.strip()] for block in blocks if block.strip()]


def answer_lines(answer: str) -> list[str]:
    return [line for block in paragraphs(answer) for line in block]


def _narrow(candidates: list[dict], line: str) -> list[dict]:
    """The candidates the line tells apart by size, by the rest of their name, then by price."""
    def service(product: dict) -> str | None:
        match = SERVICE.search(product["name"][len(product["line"]) :].replace(product.get("size") or "", " "))
        return match.group(0) if match else None

    tests = (
        lambda p: bool(p.get("size")) and contains_words(line, p["size"]),
        lambda p: bool(service(p)) and contains_words(line, service(p)),
        lambda p: f"{p['price']:.2f}" in line,
    )
    for test in tests:
        if len(candidates) == 1:
            break
        narrowed = [p for p in candidates if test(p)]
        candidates = narrowed or candidates
    return candidates


def _cells(line: str) -> list[str] | None:
    """The cells of a Markdown table row, or None for any other line."""
    row = line.strip()
    return [cell.strip() for cell in row.strip("|").split("|")] if row.startswith("|") and row.count("|") >= 2 else None


def line_key(line: str) -> str:
    """A product line as matched in answers: without markings in parentheses, which answers often leave
    out ("Pirelli Cinturato P7 All Season Run Flat (MOExtended)" -> "pirelli cinturato p7 all season run flat")."""
    return words(re.sub(r"\([^)]*\)", " ", line))


def mentions(answer: str, seen: list[dict]) -> list[Mention]:
    """The products of `seen` that the answer names, line by line.

    A product is called out of stock by a phrase on its line, by a heading such as "Out of stock:" above
    it in the same paragraph, or in a table by an availability column saying no ("| Available | ❌ No |").
    """
    by_sku = {p["sku"]: p for p in seen}
    by_line: dict[str, list[dict]] = {}
    for product in by_sku.values():
        by_line.setdefault(line_key(product["line"]), []).append(product)

    found = []
    index = -1
    for block in paragraphs(answer):
        stock_column = None
        under_unavailable_heading = False  # "**Out of stock:**" covers the lines under it in its paragraph
        for line in block:
            index += 1
            text = f" {words(line)} "
            says_unavailable = any(phrase in plain(line) for phrase in UNAVAILABLE)
            unavailable = says_unavailable or under_unavailable_heading
            cells = _cells(line)
            if cells is None:
                stock_column = None
            elif header := [i for i, cell in enumerate(cells) if STOCK_HEADER.search(plain(cell))]:
                stock_column = header[0]
            elif stock_column is not None and stock_column < len(cells):
                unavailable = unavailable or bool(NO_STOCK_CELL.search(plain(cells[stock_column])))
            named = [key for key in by_line if f" {key} " in text]
            # "Eagle F1 Asymmetric SUV-4X4" also contains the line "Eagle F1 Asymmetric SUV": keep the longest.
            named = [key for key in named if not any(key != other and f" {key} " in f" {other} " for other in named)]
            skus = {sku for sku in SKU.findall(line) if sku in by_sku}
            for key in named:
                candidates = tuple(p["sku"] for p in _narrow(by_line[key], line))
                if not skus.intersection(candidates):
                    found.append(Mention(index, candidates, unavailable))
            found += [Mention(index, (sku,), unavailable) for sku in sorted(skus)]
            if says_unavailable and not named and not skus and plain(line).endswith(":"):
                under_unavailable_heading = True
    return found


def numbers(text: str) -> set[float]:
    return {float(n.replace(",", "")) for n in NUMBER.findall(text)}


def _price_matches(value: float, has_cents: bool, price: float) -> bool:
    for count in SETS:
        total = price * count
        if (abs(total - value) < 0.006) if has_cents else value in (math.floor(total), round(total), math.ceil(total)):
            return True
    return False


def _price_supported(value: float, has_cents: bool, candidates: list[dict], seen: list[dict]) -> bool:
    """A candidate's price (per tire, pair or set), or, written to the cent, the difference between a
    candidate's price and another shown one ("$1.93 less than the Nexen")."""
    if any(_price_matches(value, has_cents, p["price"]) for p in candidates):
        return True
    return has_cents and any(
        _price_matches(value, True, abs(p["price"] - q["price"])) for p in candidates for q in seen if q["sku"] != p["sku"]
    )


def _miles(text: str) -> int:
    text = text.replace(",", "").strip().lower()
    return round(float(text[:-1].strip()) * 1000) if text.endswith("k") else int(text)


def check_facts(answer: str, seen: list[dict], shopper_numbers: set[float]) -> list[Fact]:
    """Every checkable fact in the answer, and whether the products the agent saw support it."""
    by_sku = {p["sku"]: p for p in seen}
    found_mentions = mentions(answer, seen)
    facts = []
    index = 0
    for block in paragraphs(answer):
        owner = None  # the product the current line is about
        for line in block:
            named = {sku for m in found_mentions if m.line == index for sku in m.skus}
            names_one = len(named) == 1
            if named:
                owner = by_sku[next(iter(named))] if names_one else None
            elif _cells(line) is not None:
                owner = None  # a table row is about its own product, never the row above's
            index += 1

            for match in PRICE.finditer(line):
                if BOUND.search(plain(line[: match.start()])):
                    continue
                value, has_cents = float(match.group(1).replace(",", "") + (match.group(2) or "")), bool(match.group(2))
                # A line that names the product must quote its price; a line under it may summarize others.
                candidates = [owner] if owner and names_one else seen
                supported = value in shopper_numbers or _price_supported(value, has_cents, candidates, seen)
                facts.append(Fact("price", match.group(0), supported))
            for sku in SKU.findall(line):
                facts.append(Fact("sku", sku, sku in by_sku))
            if owner is None:
                continue
            for match in TREAD.finditer(line):
                facts.append(Fact("tread_depth", match.group(0), owner.get("treadDepth32nds") == int(match.group(1))))
            for match in WARRANTY.finditer(line):
                facts.append(Fact("warranty", match.group(0), owner.get("mileageWarrantyMiles") == _miles(match.group(1))))
            for match in UTQG.finditer(line):
                facts.append(Fact("utqg", match.group(0), owner.get("utqg") == "".join(match.groups())))
            for match in RECOMMENDATION.finditer(line):
                facts.append(Fact("recommendations", match.group(0), owner.get("recommendations") == int(match.group(1))))
    return facts


def meets(product: dict, constraints: dict) -> list[str]:
    """The constraints (as search_tires arguments) the product fails; empty when it meets all of them."""
    failed = []
    for key, value in constraints.items():
        if key == "size":
            wanted = parse_size(value)
            ok = bool(product.get("size")) and parse_size(product["size"]).key == wanted.key
            ok = ok and (not wanted.light_truck or product.get("carType") == "Light Truck")
        elif key == "max_price":
            ok = product["price"] <= value + 0.005
        elif key == "min_price":
            ok = product["price"] >= value - 0.005
        elif key == "run_flat":
            ok = bool(product.get("runFlat")) == value
        else:
            field = {"car_type": "carType"}.get(key, key)
            ok = str(product.get(field, "")).lower() == str(value).lower()
        if not ok:
            failed.append(key)
    return failed
