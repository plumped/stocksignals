# stock_analyzer/notifications.py
from django.core.mail import send_mail
from django.conf import settings
from .models import WatchList, AnalysisResult


def notify_users_about_signals():
    """Benachrichtigt Benutzer über wichtige Signale bei ihren beobachteten Aktien"""
    # Alle Watchlists abrufen
    watchlists = WatchList.objects.all().prefetch_related('stocks', 'user')

    for watchlist in watchlists:
        user = watchlist.user
        if not user.email:
            continue

        # Kaufsignale finden
        buy_signals = AnalysisResult.objects.filter(
            stock__in=watchlist.stocks.all(),
            recommendation='BUY',
            technical_score__gte=80  # Starke Kaufsignale
        ).select_related('stock')

        # Verkaufssignale finden
        sell_signals = AnalysisResult.objects.filter(
            stock__in=watchlist.stocks.all(),
            recommendation='SELL',
            technical_score__lte=20  # Starke Verkaufssignale
        ).select_related('stock')

        if buy_signals or sell_signals:
            # Email-Nachricht erstellen
            message = f"Hallo {user.username},\n\n"
            message += "Hier sind die aktuellen Trading-Signale für deine beobachteten Aktien:\n\n"

            if buy_signals:
                message += "KAUFSIGNALE:\n"
                for signal in buy_signals:
                    message += f"- {signal.stock.symbol} ({signal.stock.name}): Score {signal.technical_score}\n"
                message += "\n"

            if sell_signals:
                message += "VERKAUFSSIGNALE:\n"
                for signal in sell_signals:
                    message += f"- {signal.stock.symbol} ({signal.stock.name}): Score {signal.technical_score}\n"
                message += "\n"

            message += "Besuche die Trading-Analyzer-App für detaillierte Analysen.\n\n"
            message += "Mit freundlichen Grüßen,\nDein Trading-Analyzer-Team"

            # Email senden
            # stock_analyzer/notifications.py (Fortsetzung)
            # Email senden
            send_mail(
                subject=f"Trading-Signale für deine Watchlist: {watchlist.name}",
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=True
            )