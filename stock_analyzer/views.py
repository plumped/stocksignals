# stock_analyzer/views.py
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

from .backtesting import BacktestStrategy
from .forms import UserProfileForm
from .market_analysis import MarketAnalyzer, TraditionalAnalyzer
from .ml_models import MLPredictor, AdaptiveAnalyzer
from .models import Stock, StockData, AnalysisResult, WatchList, UserProfile, MLPrediction, MLModelMetrics
from .data_service import StockDataService
from .analysis import TechnicalAnalyzer
import csv
from django.http import HttpResponse
from django.contrib import messages
from datetime import datetime, timedelta
import yfinance as yf


def index(request):
    """Dashboard-Ansicht"""
    recent_analyses = AnalysisResult.objects.order_by('-date')[:20]

    # Top Kaufempfehlungen
    buy_recommendations = AnalysisResult.objects.filter(
        recommendation='BUY'
    ).order_by('-technical_score')[:5]

    # Top Verkaufsempfehlungen
    sell_recommendations = AnalysisResult.objects.filter(
        recommendation='SELL'
    ).order_by('technical_score')[:5]

    context = {
        'recent_analyses': recent_analyses,
        'buy_recommendations': buy_recommendations,
        'sell_recommendations': sell_recommendations
    }

    return render(request, 'stock_analyzer/index.html', context)


def stock_detail(request, symbol):
    """Detailansicht für eine bestimmte Aktie"""
    stock = get_object_or_404(Stock, symbol=symbol.upper())

    # Historische Daten abrufen
    historical_data = StockData.objects.filter(stock=stock).order_by('-date')[:90]

    # Neueste Analyse abrufen
    latest_analysis = AnalysisResult.objects.filter(stock=stock).order_by('-date').first()

    # Neueste ML-Vorhersage abrufen
    latest_ml_prediction = MLPrediction.objects.filter(stock=stock).order_by('-date').first()

    # Prüfen, ob die Aktie in einer der Watchlists des Benutzers ist
    in_watchlist = WatchList.objects.filter(user=request.user,
                                            stocks=stock).exists() if request.user.is_authenticated else False

    context = {
        'stock': stock,
        'historical_data': historical_data,
        'latest_analysis': latest_analysis,
        'ml_prediction': latest_ml_prediction,
        'in_watchlist': in_watchlist
    }

    return render(request, 'stock_analyzer/stock_detail.html', context)


@login_required
def generate_ml_prediction(request, symbol):
    """Generiert eine ML-Vorhersage für eine bestimmte Aktie"""
    try:
        # Versuchen, die Aktie zu laden
        stock = get_object_or_404(Stock, symbol=symbol.upper())

        # Prüfen, ob die Aktie genügend Daten für eine ML-Vorhersage hat
        if StockData.objects.filter(stock=stock).count() < 200:
            return JsonResponse({
                'status': 'error',
                'message': 'Nicht genügend historische Daten für eine ML-Vorhersage (mindestens 200 Tage erforderlich)'
            })

        # ML-Vorhersage generieren
        predictor = MLPredictor(symbol)
        prediction = predictor.predict()

        if prediction:
            return JsonResponse({
                'status': 'success',
                'prediction': {
                    'recommendation': prediction['recommendation'],
                    'predicted_return': prediction['predicted_return'],
                    'predicted_price': prediction['predicted_price'],
                    'confidence': prediction['confidence'],
                    'prediction_days': prediction['prediction_days']
                }
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Konnte keine ML-Vorhersage erstellen'
            })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Fehler bei der ML-Vorhersage: {str(e)}'
        })


@login_required
def analyze_stock(request, symbol):
    """Führt eine technische Analyse für eine bestimmte Aktie durch"""
    try:
        # Versuchen, die Aktie zu laden oder zu erstellen
        success, message = StockDataService.update_stock_data(symbol)

        if not success:
            return JsonResponse({'status': 'error', 'message': message})

        # Jetzt sollte die Aktie in der Datenbank sein, also können wir sie analyzieren
        try:
            # Prüfen, ob genügend Daten für ML vorhanden sind
            stock = Stock.objects.get(symbol=symbol.upper())
            has_ml_data = StockData.objects.filter(stock=stock).count() >= 200

            if has_ml_data:
                # Adaptive Analyzer verwenden (kombiniert TA und ML)
                analyzer = AdaptiveAnalyzer(symbol)
                result = analyzer.get_adaptive_score()
                analysis_result = analyzer.save_analysis_result()
            else:
                # Nur traditionelle technische Analyse verwenden
                analyzer = TechnicalAnalyzer(symbol)
                result = analyzer.calculate_technical_score()
                analysis_result = analyzer.save_analysis_result()

            return JsonResponse({
                'status': 'success',
                'score': float(analysis_result.technical_score),
                'recommendation': analysis_result.recommendation,
                'signals': result['signals'],
                'has_ml_data': has_ml_data
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f"Fehler bei der Analyse: {str(e)}"})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f"Fehler: {str(e)}"})


# stock_analyzer/views.py (Fortsetzung)
@login_required
def my_watchlists(request):
    """Zeigt die Watchlisten des Benutzers an"""
    watchlists = WatchList.objects.filter(user=request.user)

    context = {
        'watchlists': watchlists
    }

    return render(request, 'stock_analyzer/my_watchlists.html', context)


@login_required
def create_watchlist(request):
    """Erstellt eine neue Watchlist"""
    if request.method == 'POST':
        name = request.POST.get('name')

        if name:
            watchlist = WatchList.objects.create(user=request.user, name=name)
            return redirect('watchlist_detail', watchlist_id=watchlist.id)

    return redirect('my_watchlists')


@login_required
def watchlist_detail(request, watchlist_id):
    """Zeigt Details einer bestimmten Watchlist an"""
    watchlist = get_object_or_404(WatchList, id=watchlist_id, user=request.user)

    # Neueste Analysen für alle Aktien in der Watchlist abrufen
    stocks = watchlist.stocks.all()
    latest_analyses = {}

    for stock in stocks:
        analysis = AnalysisResult.objects.filter(stock=stock).order_by('-date').first()
        if analysis:
            latest_analyses[stock.id] = analysis

    context = {
        'watchlist': watchlist,
        'stocks': stocks,
        'latest_analyses': latest_analyses
    }

    return render(request, 'stock_analyzer/watchlist_detail.html', context)


@login_required
def add_to_watchlist(request):
    """Fügt eine Aktie zu einer Watchlist hinzu"""
    if request.method == 'POST':
        stock_id = request.POST.get('stock_id')
        watchlist_id = request.POST.get('watchlist_id')

        stock = get_object_or_404(Stock, id=stock_id)
        watchlist = get_object_or_404(WatchList, id=watchlist_id, user=request.user)

        watchlist.stocks.add(stock)

        return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'error', 'message': 'Ungültige Anfrage'})


@login_required
def remove_from_watchlist(request):
    """Entfernt eine Aktie aus einer Watchlist"""
    if request.method == 'POST':
        stock_id = request.POST.get('stock_id')
        watchlist_id = request.POST.get('watchlist_id')

        stock = get_object_or_404(Stock, id=stock_id)
        watchlist = get_object_or_404(WatchList, id=watchlist_id, user=request.user)

        watchlist.stocks.remove(stock)

        return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'error', 'message': 'Ungültige Anfrage'})


@login_required
def search_stocks(request):
    """Sucht nach Aktien basierend auf Symbol oder Name"""
    query = request.GET.get('q', '')

    if query:
        # Zuerst in der Datenbank suchen
        stocks = Stock.objects.filter(
            symbol__icontains=query
        ) | Stock.objects.filter(
            name__icontains=query
        )

        results = []
        for stock in stocks:
            # Letzte Analyse abrufen, falls vorhanden
            latest_analysis = AnalysisResult.objects.filter(stock=stock).order_by('-date').first()

            stock_data = {
                'id': stock.id,
                'symbol': stock.symbol,
                'name': stock.name,
                'sector': stock.sector
            }

            if latest_analysis:
                stock_data.update({
                    'score': float(latest_analysis.technical_score),
                    'recommendation': latest_analysis.recommendation,
                    'analysis_date': latest_analysis.date.strftime('%Y-%m-%d')
                })

            results.append(stock_data)

        # Wenn keine lokalen Ergebnisse gefunden wurden, versuchen wir es bei Yahoo Finance
        if not results and len(query) >= 2:
            try:
                # Nach dem Symbol bei Yahoo Finance suchen
                ticker = yf.Ticker(query.upper())
                if hasattr(ticker, 'info') and 'longName' in ticker.info:
                    # Aktie existiert bei Yahoo Finance
                    # Wir erstellen sie nicht sofort, sondern zeigen sie nur als Suchergebnis an
                    stock_data = {
                        'id': 0,  # Temporäre ID
                        'symbol': query.upper(),
                        'name': ticker.info.get('longName', query.upper()),
                        'sector': ticker.info.get('sector', 'Unbekannt'),
                        'from_yahoo': True  # Markieren, dass dies ein Yahoo-Ergebnis ist
                    }
                    results.append(stock_data)
            except Exception as e:
                print(f"Fehler bei der Yahoo-Suche: {str(e)}")

        return JsonResponse({'results': results})

    return JsonResponse({'results': []})


def api_stock_data(request, symbol):
    """API-Endpunkt für Kurs- und Indikator-Daten"""
    try:
        stock = Stock.objects.get(symbol=symbol.upper())

        analyzer = TechnicalAnalyzer(stock_symbol=symbol)
        analyzer.calculate_indicators()

        df = analyzer.df  # Das berechnete DataFrame

        # Preis-Daten
        price_data = []
        for _, row in df.iterrows():
            price_data.append({
                'date': row['date'].isoformat() if isinstance(row['date'], datetime) else str(row['date']),
                'open_price': float(row['open_price']),
                'high_price': float(row['high_price']),
                'low_price': float(row['low_price']),
                'close_price': float(row['close_price']),
                'volume': int(row['volume']) if not pd.isna(row['volume']) else 0
            })

        # Indikator-Daten - ALLE verfügbaren Indikatoren einschließen
        indicators = {
            'rsi': df['rsi'].dropna().tolist() if 'rsi' in df.columns else [],
            'macd': df['macd'].dropna().tolist() if 'macd' in df.columns else [],
            'macd_signal': df['macd_signal'].dropna().tolist() if 'macd_signal' in df.columns else [],
            'macd_histogram': df['macd_histogram'].dropna().tolist() if 'macd_histogram' in df.columns else [],
            'sma_20': df['sma_20'].dropna().tolist() if 'sma_20' in df.columns else [],
            'sma_50': df['sma_50'].dropna().tolist() if 'sma_50' in df.columns else [],
            'sma_200': df['sma_200'].dropna().tolist() if 'sma_200' in df.columns else [],
            'bollinger_upper': df['bollinger_upper'].dropna().tolist() if 'bollinger_upper' in df.columns else [],
            'bollinger_lower': df['bollinger_lower'].dropna().tolist() if 'bollinger_lower' in df.columns else [],
            'stoch_k': df['stoch_k'].dropna().tolist() if 'stoch_k' in df.columns else [],
            'stoch_d': df['stoch_d'].dropna().tolist() if 'stoch_d' in df.columns else [],
            'adx': df['adx'].dropna().tolist() if 'adx' in df.columns else [],
            '+di': df['+di'].dropna().tolist() if '+di' in df.columns else [],
            '-di': df['-di'].dropna().tolist() if '-di' in df.columns else [],
            'obv': df['obv'].dropna().tolist() if 'obv' in df.columns else [],
            'atr': df['atr'].dropna().tolist() if 'atr' in df.columns else [],
            'roc': df['roc'].dropna().tolist() if 'roc' in df.columns else [],
            'psar': df['psar'].dropna().tolist() if 'psar' in df.columns else [],
            # Ichimoku-Komponenten
            'tenkan_sen': df['tenkan_sen'].dropna().tolist() if 'tenkan_sen' in df.columns else [],
            'kijun_sen': df['kijun_sen'].dropna().tolist() if 'kijun_sen' in df.columns else [],
            'senkou_span_a': df['senkou_span_a'].dropna().tolist() if 'senkou_span_a' in df.columns else [],
            'senkou_span_b': df['senkou_span_b'].dropna().tolist() if 'senkou_span_b' in df.columns else [],
            'chikou_span': df['chikou_span'].dropna().tolist() if 'chikou_span' in df.columns else []
        }

        return JsonResponse({
            'symbol': stock.symbol,
            'name': stock.name,
            'price_data': price_data,
            'indicators': indicators
        })
    except Stock.DoesNotExist:
        return JsonResponse({'error': 'Aktie nicht gefunden'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)



@login_required
def batch_analyze(request):
    """Analysiert mehrere Aktien gleichzeitig"""
    if request.method == 'POST':
        symbols = request.POST.getlist('symbols')
        results = {}

        for symbol in symbols:
            try:
                # Daten aktualisieren
                StockDataService.update_stock_data(symbol)

                # Aktie laden
                stock = Stock.objects.get(symbol=symbol.upper())
                has_ml_data = StockData.objects.filter(stock=stock).count() >= 200

                # Analyse durchführen (technisch oder adaptiv)
                if has_ml_data:
                    analyzer = AdaptiveAnalyzer(symbol)
                    result = analyzer.get_adaptive_score()
                    analysis_result = analyzer.save_analysis_result()
                else:
                    analyzer = TechnicalAnalyzer(symbol)
                    result = analyzer.calculate_technical_score()
                    analysis_result = analyzer.save_analysis_result()

                results[symbol] = {
                    'success': True,
                    'score': float(analysis_result.technical_score),
                    'recommendation': analysis_result.recommendation,
                    'confluence': result.get('confluence_score', None)
                }
            except Exception as e:
                results[symbol] = {
                    'success': False,
                    'error': str(e)
                }

        return JsonResponse({'results': results})

    return JsonResponse({'error': 'Ungültige Anfrage'}, status=400)



@login_required
def export_stock_data(request, symbol):
    """Exportiert historische Daten einer Aktie als CSV"""
    stock = get_object_or_404(Stock, symbol=symbol.upper())
    historical_data = StockData.objects.filter(stock=stock).order_by('date')

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{stock.symbol}_historical_data.csv"'

    writer = csv.writer(response)
    writer.writerow(['Datum', 'Eröffnung', 'Hoch', 'Tief', 'Schluss', 'Volumen'])

    for data in historical_data:
        writer.writerow([
            data.date.strftime('%Y-%m-%d'),
            data.open_price,
            data.high_price,
            data.low_price,
            data.close_price,
            data.volume
        ])

    return response


@login_required
def export_analysis_results(request, symbol):
    """Exportiert Analyseergebnisse einer Aktie als CSV"""
    stock = get_object_or_404(Stock, symbol=symbol.upper())
    analysis_results = AnalysisResult.objects.filter(stock=stock).order_by('date')

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{stock.symbol}_analysis_results.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Datum', 'Technischer Score', 'Empfehlung',
        'RSI', 'MACD', 'MACD Signal',
        'SMA 20', 'SMA 50', 'SMA 200'
    ])

    for result in analysis_results:
        writer.writerow([
            result.date.strftime('%Y-%m-%d'),
            result.technical_score,
            result.recommendation,
            result.rsi_value,
            result.macd_value,
            result.macd_signal,
            result.sma_20,
            result.sma_50,
            result.sma_200
        ])

    return response


@login_required
def export_watchlist(request, watchlist_id):
    """Exportiert eine Watchlist mit aktuellen Analyseergebnissen als CSV"""
    watchlist = get_object_or_404(WatchList, id=watchlist_id, user=request.user)
    stocks = watchlist.stocks.all()

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{watchlist.name}_watchlist.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Symbol', 'Name', 'Sektor',
        'Letzter Schlusskurs', 'Technischer Score', 'Empfehlung',
        'RSI', 'MACD', 'SMA 20', 'SMA 50', 'SMA 200'
    ])

    for stock in stocks:
        # Neueste Analyse abrufen
        analysis = AnalysisResult.objects.filter(stock=stock).order_by('-date').first()
        # Letzten Schlusskurs abrufen
        last_price = StockData.objects.filter(stock=stock).order_by('-date').values_list('close_price',
                                                                                         flat=True).first()

        row = [
            stock.symbol,
            stock.name,
            stock.sector or '-'
        ]

        if last_price:
            row.append(last_price)
        else:
            row.append('-')

        if analysis:
            row.extend([
                analysis.technical_score,
                analysis.recommendation,
                analysis.rsi_value,
                analysis.macd_value,
                analysis.sma_20,
                analysis.sma_50,
                analysis.sma_200
            ])
        else:
            row.extend(['-', '-', '-', '-', '-', '-', '-'])

        writer.writerow(row)

    return response


# stock_analyzer/views.py (Weitere Views für Benutzereinstellungen)
@login_required
def user_profile_settings(request):
    """Benutzereinstellungen bearbeiten"""
    profile, created = UserProfile.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Einstellungen erfolgreich gespeichert.")
            return redirect('user_profile_settings')
    else:
        form = UserProfileForm(instance=profile)

    context = {
        'form': form
    }

    return render(request, 'stock_analyzer/user_profile_settings.html', context)



@login_required
def run_backtest(request, symbol):
    """Führt einen Backtest für eine bestimmte Aktie durch"""
    stock = get_object_or_404(Stock, symbol=symbol.upper())

    # Standard-Zeitraum: Letztes Jahr
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)

    if request.method == 'POST':
        start_date_str = request.POST.get('start_date')
        end_date_str = request.POST.get('end_date')
        initial_capital = float(request.POST.get('initial_capital', 10000))

        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        backtest = BacktestStrategy(symbol, start_date, end_date, initial_capital)
        results = backtest.run_backtest()

        context = {
            'stock': stock,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'results': results
        }

        return render(request, 'stock_analyzer/backtest_results.html', context)

    # Anfangsformular anzeigen
    context = {
        'stock': stock,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': 10000
    }

    return render(request, 'stock_analyzer/backtest_form.html', context)


def market_overview(request):
    """Marktübersicht mit Sektoren und Marktbreite"""
    # Sektorperformance der letzten 30 Tage berechnen
    sector_performance = MarketAnalyzer.sector_performance(days=30)

    # Marktbreite analysieren
    market_breadth = MarketAnalyzer.market_breadth()

    # Top-Performer und Underperformer der letzten Woche
    one_week_ago = datetime.now().date() - timedelta(days=7)

    top_performers = []
    worst_performers = []

    for stock in Stock.objects.all():
        try:
            # Hier ist der Fehler - wir holen die Daten und ordnen sie in einem Schritt
            recent_data = list(StockData.objects.filter(
                stock=stock,
                date__gte=one_week_ago
            ).order_by('date'))  # Konvertiere zu einer Liste, um Slicing-Probleme zu vermeiden

            if len(recent_data) >= 2:
                first_price = float(recent_data[0].close_price)
                last_price = float(recent_data[-1].close_price)

                if first_price > 0:
                    performance = (last_price - first_price) / first_price * 100

                    stock_info = {
                        'symbol': stock.symbol,
                        'name': stock.name,
                        'sector': stock.sector,
                        'performance': performance
                    }

                    if len(top_performers) < 10 and performance > 0:
                        top_performers.append(stock_info)
                        top_performers.sort(key=lambda x: x['performance'], reverse=True)
                    elif top_performers and performance > top_performers[-1]['performance']:
                        top_performers.append(stock_info)
                        top_performers.sort(key=lambda x: x['performance'], reverse=True)
                        top_performers = top_performers[:10]

                    if len(worst_performers) < 10 and performance < 0:
                        worst_performers.append(stock_info)
                        worst_performers.sort(key=lambda x: x['performance'])
                    elif worst_performers and performance < worst_performers[-1]['performance']:
                        worst_performers.append(stock_info)
                        worst_performers.sort(key=lambda x: x['performance'])
                        worst_performers = worst_performers[:10]
        except Exception as e:
            # Protokolliere den Fehler für die Fehlerbehebung
            print(f"Fehler bei der Berechnung der Performance für {stock.symbol}: {str(e)}")
            continue

    context = {
        'sector_performance': sector_performance,
        'market_breadth': market_breadth,
        'top_performers': top_performers,
        'worst_performers': worst_performers
    }

    return render(request, 'stock_analyzer/market_overview.html', context)


@login_required
def correlation_analysis(request):
    """Korrelationsanalyse zwischen ausgewählten Aktien"""
    # Standardmäßig einige Indizes/ETFs für die Analyse
    default_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VGK', 'EEM']
    symbols = request.GET.getlist('symbols', default_symbols)

    # Timeframe
    days = int(request.GET.get('days', 90))

    # Korrelationsmatrix berechnen
    correlation_matrix = None
    if len(symbols) >= 2:
        try:
            correlation_matrix = MarketAnalyzer.calculate_correlations(symbols, days)
        except Exception as e:
            messages.error(request, f"Fehler bei der Korrelationsanalyse: {str(e)}")

    # Alle Aktien für die Dropdown-Liste
    all_stocks = Stock.objects.all().order_by('symbol')

    context = {
        'correlation_matrix': correlation_matrix,
        'selected_symbols': symbols,
        'days': days,
        'all_stocks': all_stocks
    }

    return render(request, 'stock_analyzer/correlation_analysis.html', context)


@login_required
def delete_watchlist(request, watchlist_id):
    """Löscht eine Watchlist"""
    if request.method == 'POST':
        watchlist = get_object_or_404(WatchList, id=watchlist_id, user=request.user)
        watchlist.delete()
        return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'error', 'message': 'Ungültige Anfrage'}, status=400)


def advanced_indicators(request, symbol):
    """Zeigt fortgeschrittene technische Indikatoren für eine bestimmte Aktie an"""
    stock = get_object_or_404(Stock, symbol=symbol.upper())

    # Analyzer mit erweiterten Indikatoren erstellen
    analyzer = TechnicalAnalyzer(symbol)

    # Berechne Standardindikatoren und dann die erweiterten Indikatoren
    analyzer.calculate_indicators(include_advanced=True)

    # Für die Ansicht benötigte Daten abrufen
    last_date = analyzer.df['date'].iloc[-1]
    latest_close = float(analyzer.df['close_price'].iloc[-1])

    # Erkannte Chartmuster finden - VERBESSERTE VERSION
    # Alle erkannten Muster für die Tabelle
    all_detected_patterns = []
    pattern_columns = [col for col in analyzer.df.columns if col.startswith('pattern_')]

    if pattern_columns:
        # Hier ist der wichtige Teil: Prüfe die letzten 5-10 Tage auf Muster
        recent_days = min(10, len(analyzer.df))
        recent_data = analyzer.df.tail(recent_days)

        for _, row in recent_data.iterrows():
            for col in pattern_columns:
                if row[col] == 1:  # Muster erkannt
                    pattern_name = col.replace('pattern_', '').replace('_', ' ').title()

                    # Signaltyp und Beschreibung basierend auf dem Mustertyp
                    signal = 'HOLD'
                    description = 'Mögliche Trendwende.'
                    reliability = 70

                    if 'double_top' in col or 'head_shoulders' in col:
                        signal = 'SELL'
                        description = 'Potentielles Trendumkehrmuster nach oben. Verkaufssignal.'
                        reliability = 75
                    elif 'double_bottom' in col or 'inv_head_shoulders' in col:
                        signal = 'BUY'
                        description = 'Potentielles Trendumkehrmuster nach unten. Kaufsignal.'
                        reliability = 75
                    elif 'triangle_ascending' in col:
                        signal = 'BUY'
                        description = 'Aufsteigendes Dreieck. Fortsetzung des Aufwärtstrends wahrscheinlich.'
                        reliability = 80
                    elif 'triangle_descending' in col:
                        signal = 'SELL'
                        description = 'Absteigendes Dreieck. Fortsetzung des Abwärtstrends wahrscheinlich.'
                        reliability = 80
                    elif 'flag_bullish' in col:
                        signal = 'BUY'
                        description = 'Bullische Flagge. Fortsetzung des Aufwärtstrends wahrscheinlich.'
                        reliability = 85
                    elif 'flag_bearish' in col:
                        signal = 'SELL'
                        description = 'Bärische Flagge. Fortsetzung des Abwärtstrends wahrscheinlich.'
                        reliability = 85

                    # Füge das erkannte Muster zur Liste hinzu
                    all_detected_patterns.append({
                        'name': pattern_name,
                        'signal': signal,
                        'description': description,
                        'reliability': reliability,
                        'date': row['date']
                    })



    # Suche nach Mustern in den letzten 30 Tagen
    last_30_days = analyzer.df.tail(30)
    for _, row in last_30_days.iterrows():
        for col in pattern_columns:
            if row[col] == 1:
                pattern_name = col.replace('pattern_', '').replace('_', ' ').title()

                # Signaltyp und Beschreibung bestimmen (wie oben)
                signal = 'HOLD'
                description = 'Mögliche Trendwende.'
                reliability = 70

                if 'double_top' in col or 'head_shoulders' in col:
                    signal = 'SELL'
                    description = 'Potentielles Trendumkehrmuster nach oben.'
                    reliability = 75
                elif 'double_bottom' in col or 'inv_head_shoulders' in col:
                    signal = 'BUY'
                    description = 'Potentielles Trendumkehrmuster nach unten.'
                    reliability = 75
                elif 'triangle_ascending' in col:
                    signal = 'BUY'
                    description = 'Aufsteigendes Dreieck.'
                    reliability = 80
                elif 'triangle_descending' in col:
                    signal = 'SELL'
                    description = 'Absteigendes Dreieck.'
                    reliability = 80
                elif 'flag_bullish' in col:
                    signal = 'BUY'
                    description = 'Bullische Flagge.'
                    reliability = 85
                elif 'flag_bearish' in col:
                    signal = 'SELL'
                    description = 'Bärische Flagge.'
                    reliability = 85

                all_detected_patterns.append({
                    'name': pattern_name,
                    'signal': signal,
                    'description': description,
                    'reliability': reliability,
                    'date': row['date']
                })

    # SuperTrend-Signal
    supertrend_signal = 'HOLD'
    if 'supertrend_direction' in analyzer.df.columns:
        last_direction = analyzer.df['supertrend_direction'].iloc[-1]
        if last_direction > 0:
            supertrend_signal = 'BUY'
        elif last_direction < 0:
            supertrend_signal = 'SELL'

    # Elliott Wave Analyse
    elliott_wave_analysis = None
    if 'elliott_wave_point' in analyzer.df.columns and 'elliott_wave_pattern' in analyzer.df.columns:
        # Vereinfachte Analyse basierend auf den letzten identifizierten Punkten
        last_points = analyzer.df[analyzer.df['elliott_wave_point'] != 0].tail(5)

        if not last_points.empty:
            # Zähle abwechselnde Hoch- und Tiefpunkte
            point_sequence = last_points['elliott_wave_point'].tolist()

            # Bestimme mögliche Elliott-Wellen-Struktur
            wave_description = "Unbekannte Wave-Position"
            current_wave = "Nicht identifizierbar"

            # Vereinfachte Analyse basierend auf den letzten Punkten
            if len(point_sequence) >= 5:
                # Prüfe auf 5-Wellen-Impuls oder 3-Wellen-Korrektur
                up_points = sum(1 for p in point_sequence if p > 0)
                down_points = sum(1 for p in point_sequence if p < 0)

                if up_points == 3 and down_points == 2:
                    current_wave = "Wahrscheinlich 5-Wellen-Impuls (aufwärts)"
                    wave_description = "Impulsphase in Aufwärtsrichtung. Potentielles Kaufsignal."
                elif up_points == 2 and down_points == 3:
                    current_wave = "Wahrscheinlich 5-Wellen-Impuls (abwärts)"
                    wave_description = "Impulsphase in Abwärtsrichtung. Potentielles Verkaufssignal."
                elif up_points == 2 and down_points == 1:
                    current_wave = "Möglicherweise A-B-C Korrektur"
                    wave_description = "Korrekturphase könnte abgeschlossen sein."

            elliott_wave_analysis = {
                'current_wave': current_wave,
                'description': wave_description
            }

    # Fibonacci-Levels
    fibonacci_levels = []

    # Bestimme, ob wir uns in einem Aufwärts- oder Abwärtstrend befinden
    trend_up = latest_close > analyzer.df['sma_50'].iloc[-1] if 'sma_50' in analyzer.df.columns else True

    # Suche nach Fibonacci-Retracement-Columns
    fib_prefix = 'fib_up_' if not trend_up else 'fib_down_'
    fib_columns = [col for col in analyzer.df.columns if col.startswith(fib_prefix)]

    if fib_columns:
        # Fibonacci-Level-Namen und deren Prozentwerte
        fib_mappings = {
            '0': 0,
            '236': 0.236,
            '382': 0.382,
            '500': 0.5,
            '618': 0.618,
            '786': 0.786,
            '1000': 1.0
        }

        # Fibonacci-Farben
        fib_colors = {
            '0': '#6c757d',
            '236': '#28a745',
            '382': '#17a2b8',
            '500': '#fd7e14',
            '618': '#dc3545',
            '786': '#6610f2',
            '1000': '#343a40'
        }

        for suffix, value in fib_mappings.items():
            col_name = f"{fib_prefix}{suffix}"
            if col_name in analyzer.df.columns:
                fib_value = float(analyzer.df[col_name].iloc[-1])
                # Prüfe auf NaN oder ungültige Werte
                if not pd.isna(fib_value) and fib_value > 0:
                    fibonacci_levels.append({
                        'name': f"Fibonacci {value}",
                        'value': fib_value,
                        'color': fib_colors.get(suffix, '#000000')
                    })
                else:
                    # Fallback-Wert basierend auf aktuellem Preis und Level
                    if trend_up:
                        fallback_value = latest_close * (1 - value * 0.1)
                    else:
                        fallback_value = latest_close * (1 + value * 0.1)

                    fibonacci_levels.append({
                        'name': f"Fibonacci {value}",
                        'value': fallback_value,
                        'color': fib_colors.get(suffix, '#000000')
                    })

    # VWAP-Daten
    vwap_data = None
    if 'vwap' in analyzer.df.columns:
        current_vwap = float(analyzer.df['vwap'].iloc[-1])
        # Prüfe auf NaN
        if pd.isna(current_vwap):
            # Fallback: Verwende den aktuellen Preis als VWAP
            current_vwap = latest_close

        difference = ((latest_close - current_vwap) / current_vwap) * 100

        vwap_data = {
            'current': current_vwap,
            'difference': abs(difference)
        }

    context = {
        'stock': stock,
        'current_price': latest_close,
        'all_detected_patterns': all_detected_patterns,
        'supertrend_signal': supertrend_signal,
        'elliott_wave_analysis': elliott_wave_analysis,
        'fibonacci_levels': fibonacci_levels,
        'vwap_data': vwap_data
    }

    return render(request, 'stock_analyzer/advanced_indicators.html', context)


def api_advanced_indicators(request, symbol):
    """API-Endpunkt für fortgeschrittene Indikator-Daten"""
    try:
        stock = Stock.objects.get(symbol=symbol.upper())

        # Analyzer mit erweiterten Indikatoren erstellen
        analyzer = TechnicalAnalyzer(symbol)
        analyzer.calculate_indicators(include_advanced=True)

        # Preis-Daten
        price_data = []
        for _, row in analyzer.df.iterrows():
            price_data.append({
                'date': row['date'].isoformat() if isinstance(row['date'], datetime) else str(row['date']),
                'open_price': float(row['open_price']),
                'high_price': float(row['high_price']),
                'low_price': float(row['low_price']),
                'close_price': float(row['close_price']),
                'volume': int(row['volume']) if not pd.isna(row['volume']) else 0
            })

        # Extrahiere alle fortgeschrittenen Indikatoren
        advanced_indicators = {}

        # Heikin-Ashi Daten
        for ha_col in ['ha_open', 'ha_high', 'ha_low', 'ha_close', 'ha_trend']:
            if ha_col in analyzer.df.columns:
                advanced_indicators[ha_col] = analyzer.df[ha_col].fillna(0).tolist()

        # SuperTrend Daten
        for st_col in ['supertrend', 'supertrend_direction']:
            if st_col in analyzer.df.columns:
                advanced_indicators[st_col] = analyzer.df[st_col].fillna(0).tolist()

        # Elliott Wave Daten
        for ew_col in ['elliott_wave_point', 'elliott_wave_pattern']:
            if ew_col in analyzer.df.columns:
                advanced_indicators[ew_col] = analyzer.df[ew_col].fillna(0).tolist()

        # Fibonacci Levels
        fib_columns = [col for col in analyzer.df.columns if col.startswith('fib_')]
        for fib_col in fib_columns:
            if fib_col in analyzer.df.columns:
                advanced_indicators[fib_col] = analyzer.df[fib_col].fillna(0).tolist()

        # Chart-Muster
        pattern_columns = [col for col in analyzer.df.columns if col.startswith('pattern_')]
        for pattern_col in pattern_columns:
            if pattern_col in analyzer.df.columns:
                advanced_indicators[pattern_col] = analyzer.df[pattern_col].fillna(0).tolist()

        # VWAP
        if 'vwap' in analyzer.df.columns:
            advanced_indicators['vwap'] = analyzer.df['vwap'].fillna(0).tolist()

        # Standardindikatoren auch einbeziehen
        standard_indicators = {
            'rsi': analyzer.df['rsi'].fillna(0).tolist() if 'rsi' in analyzer.df.columns else [],
            'macd': analyzer.df['macd'].fillna(0).tolist() if 'macd' in analyzer.df.columns else [],
            'macd_signal': analyzer.df['macd_signal'].fillna(
                0).tolist() if 'macd_signal' in analyzer.df.columns else [],
            'sma_20': analyzer.df['sma_20'].fillna(0).tolist() if 'sma_20' in analyzer.df.columns else [],
            'sma_50': analyzer.df['sma_50'].fillna(0).tolist() if 'sma_50' in analyzer.df.columns else [],
            'sma_200': analyzer.df['sma_200'].fillna(0).tolist() if 'sma_200' in analyzer.df.columns else []
        }

        # Alle Indikatoren zusammenführen
        all_indicators = {**standard_indicators, **advanced_indicators}

        return JsonResponse({
            'symbol': stock.symbol,
            'name': stock.name,
            'price_data': price_data,
            'indicators': all_indicators
        })
    except Stock.DoesNotExist:
        return JsonResponse({'error': 'Aktie nicht gefunden'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def evaluate_ml_model(request, symbol):
    """Evaluiert das ML-Modell für eine bestimmte Aktie"""
    try:
        # Prüfen, ob die Aktie existiert
        stock = get_object_or_404(Stock, symbol=symbol.upper())

        # Modell evaluieren
        predictor = MLPredictor(symbol)
        performance = predictor.evaluate_model_performance()

        if performance:
            return JsonResponse({
                'status': 'success',
                'performance': performance
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Konnte das Modell nicht evaluieren'
            })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Fehler bei der Modellevaluation: {str(e)}'
        })


@login_required
def ml_dashboard(request):
    """Machine Learning Dashboard mit Übersicht über Vorhersagen und Modellleistung"""
    from django.db.models import Avg, Count
    from datetime import datetime
    import os
    import json
    from .models import MLModelMetrics

    # Anzahl der Aktien mit ML-Modellen (Datei auf Festplatte prüfen)
    models_dir = 'ml_models'
    model_files = []
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

    model_count = len(model_files) // 2  # Jede Aktie hat zwei Modelle (price und signal)

    # Aktuelle ML-Statistiken
    prediction_count = MLPrediction.objects.count()
    avg_accuracy_obj = MLModelMetrics.objects.aggregate(avg_accuracy=Avg('accuracy'))
    avg_accuracy = round((avg_accuracy_obj['avg_accuracy'] or 0) * 100, 1)  # Auf Prozent umrechnen und runden
    last_update = MLPrediction.objects.order_by('-date').first()
    last_update_date = last_update.date if last_update else datetime.now().date()

    ml_stats = {
        'model_count': model_count,
        'prediction_count': prediction_count,
        'avg_accuracy': avg_accuracy,
        'last_update': last_update_date
    }

    # Top Kauf- und Verkaufsempfehlungen abrufen
    top_buy_predictions = MLPrediction.objects.filter(
        recommendation='BUY',
        confidence__gte=0.6
    ).order_by('-predicted_return')[:10]

    top_sell_predictions = MLPrediction.objects.filter(
        recommendation='SELL',
        confidence__gte=0.6
    ).order_by('predicted_return')[:10]

    # ML-Performance abrufen
    metrics = MLModelMetrics.objects.order_by('-date')[:10]

    symbols = []
    accuracies = []

    for metric in metrics:
        if metric.accuracy is not None:
            symbols.append(metric.stock.symbol)
            accuracies.append(round(metric.accuracy * 100, 2))

    # Traditionelle Performance - NEU
    traditional_performance = TraditionalAnalyzer.evaluate_traditional_performance(symbols[0]) if symbols else {
        'accuracy': 65,
        'return': 70,
        'speed': 85,
        'adaptability': 50,
        'robustness': 80
    }

    performance_data = {
        'symbols': json.dumps(symbols),
        'accuracy': json.dumps(accuracies),
        'traditional': json.dumps([
            float(traditional_performance['accuracy']),
            float(traditional_performance['return']),
            float(traditional_performance['speed']),
            float(traditional_performance['adaptability']),
            float(traditional_performance['robustness'])
        ]),
        'traditional_final': round((
                float(traditional_performance['accuracy']) * 0.3 +
                float(traditional_performance['return']) * 0.25 +
                float(traditional_performance['speed']) * 0.2 +
                float(traditional_performance['adaptability']) * 0.15 +
                float(traditional_performance['robustness']) * 0.1
        ), 1)
    }

    # Aktien mit genug Daten für ML abrufen
    stocks_with_data = Stock.objects.annotate(
        data_count=Count('historical_data')
    ).filter(data_count__gte=200).order_by('symbol')

    context = {
        'ml_stats': ml_stats,
        'top_buy_predictions': top_buy_predictions,
        'top_sell_predictions': top_sell_predictions,
        'performance_data': performance_data,
        'stocks_with_data': stocks_with_data
    }

    return render(request, 'stock_analyzer/ml_dashboard.html', context)






@login_required
def batch_ml_predictions_view(request):
    """Django-View für Batch-ML-Vorhersagen"""
    from .ml_models import batch_ml_predictions as run_batch_predictions  # <- wichtig!

    symbols_param = request.GET.get('symbols', '')
    retrain = request.GET.get('retrain', 'false').lower() == 'true'

    if symbols_param == 'watchlist':
        watchlist_stocks = WatchList.objects.filter(
            user=request.user
        ).values_list('stocks__symbol', flat=True).distinct()

        symbols = list(filter(None, watchlist_stocks))
    elif symbols_param:
        symbols = [symbols_param]
    else:
        symbols = None

    try:
        results = run_batch_predictions(symbols, retrain)

        return JsonResponse({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Fehler bei der Batch-Verarbeitung: {str(e)}'
        })
