from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .io import Record


_LEGAL_SUFFIX_PHRASES = (
    "limited liability company",
    "private limited",
    "pvt ltd",
    "incorporated",
    "corporation",
    "company limited",
    "company",
    "limited",
    "private",
    "proprietary",
    "corporation",
    "llc",
    "llp",
    "l l p",
    "inc",
    "ltd",
    "pvt",
    "plc",
    "sarl",
    "sas",
    "gmbh",
    "ag",
    "bv",
    "nv",
    "pte",
    "sa",
    "co",
    "corp",
)
_STRONG_LEGAL_TOKENS = {
    "llc",
    "llp",
    "inc",
    "ltd",
    "limited",
    "pvt",
    "private",
    "corp",
    "corporation",
    "plc",
    "sarl",
    "sas",
    "gmbh",
    "ag",
    "bv",
    "nv",
    "pte",
    "sa",
}
_ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "ter": "terrace",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}
_STREET_TERMS = set(_ADDRESS_ABBREVIATIONS.values()) | {
    "boulevard",
    "road",
    "street",
    "avenue",
    "drive",
    "lane",
    "court",
    "place",
    "square",
    "highway",
    "parkway",
    "circle",
    "terrace",
    "rue",
    "avenue",
    "via",
    "viale",
    "strasse",
    "straße",
    "gasse",
    "platz",
}
_NON_CITY_TOKENS = {
    "near",
    "opposite",
    "opposite to",
    "beside",
    "behind",
    "front",
    "adjacent",
    "next",
    "floor",
    "unit",
    "apartment",
    "flat",
    "suite",
    "ste",
    "building",
    "bldg",
    "landmark",
    "atm",
    "post",
    "office",
    "po",
    "box",
}
_NULL_TOKENS = {"", "null", "none", "n/a", "na", "-", "--"}
_PUNCTUATION_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_SPACES_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+")
_POSTAL_RE = re.compile(r"(?<!\d)\d{5,6}(?!\d)")


@dataclass(frozen=True)
class NameNormalization:
    raw: str
    clean: str
    folded: str
    core: str
    core_folded: str
    tokens: tuple[str, ...]
    core_tokens: tuple[str, ...]
    suffix: tuple[str, ...]
    token_sort: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class AddressNormalization:
    raw: str
    clean: str
    folded: str
    tokens: tuple[str, ...]
    numbers: tuple[str, ...]
    house_number: str | None
    postal_code: str | None
    city: str | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedRecord:
    raw: Record
    name: NameNormalization
    address: AddressNormalization
    country: str
    flags: tuple[str, ...]


def _base_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold().replace("&", " and ")
    text = _PUNCTUATION_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def _fold_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(character for character in text if not unicodedata.combining(character))
    return _SPACES_RE.sub(" ", text).strip()


def _legal_suffix_end(tokens: list[str]) -> tuple[str, ...]:
    suffix: list[str] = []
    changed = True
    while changed and tokens:
        changed = False
        for phrase in _LEGAL_SUFFIX_PHRASES:
            phrase_tokens = phrase.split()
            phrase_length = len(phrase_tokens)
            if len(tokens) >= phrase_length and tuple(tokens[-phrase_length:]) == tuple(phrase_tokens):
                suffix = phrase_tokens + suffix
                del tokens[-phrase_length:]
                changed = True
                break
    return tuple(suffix)


def normalize_name_detail(value: str) -> NameNormalization:
    raw = str(value or "")
    clean = _base_text(raw)
    tokens = clean.split() if clean else []
    suffix = list(_legal_suffix_end(tokens))
    filtered_tokens: list[str] = []
    for token in tokens:
        if token in _STRONG_LEGAL_TOKENS:
            suffix.append(token)
        else:
            filtered_tokens.append(token)
    tokens[:] = filtered_tokens
    core = " ".join(tokens)
    folded = _fold_text(clean)
    core_folded = _fold_text(core)
    token_sort = " ".join(sorted(clean.split())) if clean else ""
    flags: list[str] = []
    if not clean:
        flags.append("blank_name")
    elif _base_text(raw) in _NULL_TOKENS:
        flags.append("null_name")
    if len(suffix) > 1 and len(set(suffix)) < len(suffix):
        flags.append("multiple_legal_suffixes")
    return NameNormalization(
        raw=raw,
        clean=clean,
        folded=folded,
        core=core,
        core_folded=core_folded,
        tokens=tuple(clean.split()),
        core_tokens=tuple(tokens),
        suffix=tuple(suffix),
        token_sort=token_sort,
        flags=tuple(flags),
    )


def normalize_name(value: str) -> str:
    return normalize_name_detail(value).clean


def _expand_address_tokens(tokens: list[str]) -> list[str]:
    return [_ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens]


def _extract_house_number(value: str) -> str | None:
    for token in _base_text(value).split():
        compact = re.sub(r"[^a-z0-9]", "", token)
        if not compact or not any(character.isdigit() for character in compact):
            continue
        if compact[0].isdigit() or (len(compact) > 1 and compact[0].isalpha() and compact[1].isdigit()):
            return compact
    return None


def _extract_city(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens[:-1]):
        if token in _STREET_TERMS:
            for candidate in tokens[index + 1 :]:
                if candidate in _NON_CITY_TOKENS or candidate.isdigit():
                    continue
                return candidate
    return None


def normalize_address(value: str) -> AddressNormalization:
    raw = str(value or "")
    base_tokens = _base_text(raw).split() if _base_text(raw) else []
    tokens = _expand_address_tokens(base_tokens)
    clean = " ".join(tokens)
    folded = _fold_text(clean)
    numbers = tuple(_NUMBER_RE.findall(clean))
    postal_match = _POSTAL_RE.search(clean)
    flags: list[str] = []
    if not clean:
        flags.append("blank_address")
    elif clean in _NULL_TOKENS:
        flags.append("null_address")
    if clean and not any(character.isalpha() for character in clean):
        flags.append("address_without_letters")
    if clean and not numbers:
        flags.append("address_without_numbers")
    return AddressNormalization(
        raw=raw,
        clean=clean,
        folded=folded,
        tokens=tuple(tokens),
        numbers=numbers,
        house_number=_extract_house_number(raw),
        postal_code=postal_match.group(0) if postal_match else None,
        city=_extract_city(tokens),
        flags=tuple(flags),
    )


def normalize_country(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return _SPACES_RE.sub(" ", text.casefold()).strip()


def normalize_record(record: Record) -> NormalizedRecord:
    name = normalize_name_detail(record.business_name)
    address = normalize_address(record.business_address)
    country = normalize_country(record.country)
    flags = tuple(name.flags + address.flags)
    if not country:
        flags += ("blank_country",)
    return NormalizedRecord(
        raw=record,
        name=name,
        address=address,
        country=country,
        flags=flags,
    )
