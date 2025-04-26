# stock_analyzer/templatetags/stock_tags.py
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Hilfsfunktion zum Abrufen von Elementen aus einem Dictionary in Templates"""
    return dictionary.get(key)

@register.filter
def price_color(value, baseline=0):
    """Färbt einen Wert je nach Relation zum Basiswert"""
    if value > baseline:
        return mark_safe(f'<span class="text-success">{value}</span>')
    elif value < baseline:
        return mark_safe(f'<span class="text-danger">{value}</span>')
    else:
        return value

@register.filter
def percentage(value, decimals=2):
    """Formatiert einen Wert als Prozentsatz"""
    return f"{value * 100:.{decimals}f}%"

@register.filter
def recommendation_badge(recommendation):
    """Erzeugt ein farbcodiertes Badge für Empfehlungen"""
    if recommendation == 'BUY':
        return mark_safe('<span class="badge bg-success">KAUFEN</span>')
    elif recommendation == 'SELL':
        return mark_safe('<span class="badge bg-danger">VERKAUFEN</span>')
    else:
        return mark_safe('<span class="badge bg-warning">HALTEN</span>')