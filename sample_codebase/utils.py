"""
Utility functions for e-commerce billing.
"""

from typing import Optional


def format_currency(amount: Optional[float], currency_symbol: str = "$") -> str:
    """Formats amount as currency string."""
    if amount is None:
        return f"{currency_symbol}0.00"
    return f"{currency_symbol}{amount:,.2f}"


def sanitize_tier_name(tier: Optional[str]) -> str:
    """Sanitizes member tier name."""
    if not tier or not isinstance(tier, str):
        return "regular"
    return tier.strip().lower()
