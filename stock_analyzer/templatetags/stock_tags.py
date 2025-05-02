# stock_analyzer/templatetags/stock_tags.py
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def get_item(obj, attr):
    """Hilfsfunktion für den Zugriff auf dynamische Attribute oder Dictionary-Keys in Templates"""
    try:
        if isinstance(obj, dict):
            return obj.get(attr)
        elif hasattr(obj, 'loc'):  # Pandas DataFrame oder Series
            return obj.loc[attr]
        else:
            return getattr(obj, attr, None)
    except (KeyError, AttributeError, TypeError):
        return None

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

@register.filter
def getattr(obj, attr):
    """Hilfsfunktion für den Zugriff auf dynamische Attribute oder Dictionary-Keys in Templates"""
    if isinstance(obj, dict):
        return obj.get(attr)
    else:
        return getattr(obj, attr, None)

@register.filter
def sub(value, arg):
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return ''