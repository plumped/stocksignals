# stock_analyzer/templatetags/math_filters.py
from django import template
from decimal import Decimal

register = template.Library()

@register.filter
def mul(value, arg):
    """Multiplies the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return None

@register.filter
def div(value, arg):
    """Divides the value by the argument"""
    try:
        arg = float(arg)
        if arg == 0:
            return None  # Avoid division by zero
        return float(value) / arg
    except (ValueError, TypeError):
        return None