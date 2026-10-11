"""
Unit tests for sample e-commerce calculator.
Run with: python -m unittest sample_codebase/test_sample.py
"""

import unittest
from calculator import calculate_discount, calculate_final_bill


class TestCalculator(unittest.TestCase):

    def test_regular_purchase(self):
        # 2 items totaling $250 -> average $125 > 100 -> 5% discount = $12.50
        discount = calculate_discount(250.0, 2, "regular")
        self.assertEqual(discount, 12.50)

    def test_gold_member(self):
        discount = calculate_discount(100.0, 1, "gold")
        self.assertEqual(discount, 15.00)

    def test_zero_items_edge_case(self):
        """This test currently crashes with ZeroDivisionError due to the bug!"""
        discount = calculate_discount(0.0, 0, "regular")
        self.assertEqual(discount, 0.0)


if __name__ == "__main__":
    unittest.main()
