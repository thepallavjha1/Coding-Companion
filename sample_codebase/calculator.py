"""
Sample e-commerce billing calculations module.
Contains an intentional bug in calculate_discount for testing the agent.
"""

from typing import Dict, Any


def calculate_discount(total_amount: float, total_items: int, member_tier: str) -> float:
    """
    Calculates applicable discount on a cart purchase.
    
    Bug alert: If total_items is 0 (e.g. empty cart or promotional credit calculation),
    this line raises ZeroDivisionError!
    """
    # BUG: No check for total_items == 0 before division
    average_item_price = total_amount / total_items

    discount = 0.0
    if member_tier.lower() == "gold":
        discount = total_amount * 0.15
    elif member_tier.lower() == "silver":
        discount = total_amount * 0.10
    elif average_item_price > 100:
        discount = total_amount * 0.05

    return round(discount, 2)


def calculate_tax(amount: float, tax_rate: float = 0.08) -> float:
    """Calculates tax on post-discount amount."""
    if amount < 0:
        raise ValueError("Amount cannot be negative")
    return round(amount * tax_rate, 2)


def calculate_final_bill(total_amount: float, total_items: int, member_tier: str = "regular") -> Dict[str, Any]:
    """Generates the overall final bill summary."""
    discount = calculate_discount(total_amount, total_items, member_tier)
    taxable_amount = max(0.0, total_amount - discount)
    tax = calculate_tax(taxable_amount)
    grand_total = round(taxable_amount + tax, 2)

    return {
        "subtotal": total_amount,
        "discount": discount,
        "tax": tax,
        "grand_total": grand_total,
    }
