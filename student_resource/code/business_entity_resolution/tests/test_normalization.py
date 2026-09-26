import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import Record
from src.normalize import (
    normalize_address,
    normalize_country,
    normalize_name,
    normalize_name_detail,
    normalize_record,
)


class NormalizationTests(unittest.TestCase):
    def test_name_normalization_preserves_core_and_suffix(self):
        detail = normalize_name_detail("  Acme, Inc. & Sons  ")
        self.assertEqual(normalize_name("Acme, Inc."), "acme inc")
        self.assertEqual(detail.clean, "acme inc and sons")
        self.assertEqual(detail.core, "acme and sons")
        self.assertEqual(detail.suffix, ("inc",))
        self.assertEqual(detail.token_sort, "acme and inc sons")

    def test_leading_legal_suffix_and_word_order(self):
        detail = normalize_name_detail("LLC Moncada Learning Center")
        self.assertEqual(detail.core, "moncada learning center")
        self.assertEqual(normalize_name("Center Learning Moncada"), "center learning moncada")

    def test_address_abbreviation_and_house_number(self):
        detail = normalize_name_detail("Café École")
        self.assertEqual(detail.clean, "café école")
        self.assertEqual(detail.folded, "cafe ecole")
        address = normalize_address("12B Park Rd, Near SBI ATM")
        self.assertEqual(address.clean, "12b park road near sbi atm")
        self.assertEqual(address.house_number, "12b")
        self.assertIn("12", address.numbers)
        self.assertEqual(address.postal_code, None)

    def test_postal_code_and_missing_address_flags(self):
        address = normalize_address("21B Rue de Paris, 75001")
        self.assertEqual(address.postal_code, "75001")
        missing = normalize_address("")
        self.assertEqual(missing.clean, "")
        self.assertIn("blank_address", missing.flags)

    def test_country_is_open_normalized_string(self):
        self.assertEqual(normalize_country("  FRANCE "), "france")
        self.assertEqual(normalize_country("FR"), "fr")

    def test_record_keeps_raw_fields(self):
        record = Record("S1-1", "Acme Pvt Ltd", "1 Road", "US")
        normalized = normalize_record(record)
        self.assertEqual(normalized.raw.business_name, "Acme Pvt Ltd")
        self.assertEqual(normalized.name.core, "acme")
        self.assertEqual(normalized.country, "us")
        self.assertEqual(normalized.address.house_number, "1")


if __name__ == "__main__":
    unittest.main()
