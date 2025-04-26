# stock_analyzer/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

from .backtesting import BacktestStrategy
from .forms import IndicatorWeightForm, UserProfileForm
from .market_analysis import MarketAnalyzer
from .models import Stock, StockData, AnalysisResult, WatchList, UserProfile
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


@login_required
def stock_detail(request, symbol):
    """Detailansicht für eine bestimmte Aktie"""
    stock = get_object_or_404(Stock, symbol=symbol.upper())

    # Historische Daten abrufen
    historical_data = StockData.objects.filter(stock=stock).order_by('-date')[:90]

    # Neueste Analyse abrufen
    latest_analysis = AnalysisResult.objects.filter(stock=stock).order_by('-date').first()

    # Prüfen, ob die Aktie in einer der Watchlists des Benutzers ist
    in_watchlist = WatchList.objects.filter(user=request.user, stocks=stock).exists()

    context = {
        'stock': stock,
        'historical_data': historical_data,
        'latest_analysis': latest_analysis,
        'in_watchlist': in_watchlist
    }

    return render(request, 'stock_analyzer/stock_detail.html', context)


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
            analyzer = TechnicalAnalyzer(symbol)
            result = analyzer.calculate_technical_score()
            analysis_result = analyzer.save_analysis_result()

            return JsonResponse({
                'status': 'success',
                'score': float(analysis_result.technical_score),
                'recommendation': analysis_result.recommendation,
                'signals': result['signals']
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
    """API-Endpunkt für Aktiendaten (JSON)"""
    try:
        stock = Stock.objects.get(symbol=symbol.upper())

        # Kursdaten der letzten 90 Tage
        price_data = list(StockData.objects.filter(stock=stock).order_by('date')
                          .values('date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume')[:90])

        # Technische Indikatoren
        analyzer = TechnicalAnalyzer(symbol)
        analyzer.calculate_indicators()

        # Indikatorwerte für das Frontend
        indicators = {}
        for col in analyzer.df.columns:
            if col not in ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                indicators[col] = analyzer.df[col].dropna().tolist()

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

                # Analyse durchführen
                analyzer = TechnicalAnalyzer(symbol)
                analysis_result = analyzer.save_analysis_result()

                results[symbol] = {
                    'success': True,
                    'score': float(analysis_result.technical_score),
                    'recommendation': analysis_result.recommendation
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
def indicator_weights_settings(request):
    """Benutzereinstellungen für Indikatorgewichtungen"""
    profile, created = UserProfile.objects.get_or_create(user=request.user)

    # Standardwerte oder gespeicherte Werte abrufen
    initial_values = profile.custom_weights if profile.custom_weights else {
        'rsi_weight': 10,
        'macd_weight': 10,
        'sma_weight': 20,
        'bollinger_weight': 10,
        'stochastic_weight': 10,
        'adx_weight': 10,
        'ichimoku_weight': 10,
        'obv_weight': 10
    }

    if request.method == 'POST':
        form = IndicatorWeightForm(request.POST, initial=initial_values)
        if form.is_valid():
            profile.custom_weights = {
                'rsi_weight': form.cleaned_data['rsi_weight'],
                'macd_weight': form.cleaned_data['macd_weight'],
                'sma_weight': form.cleaned_data['sma_weight'],
                'bollinger_weight': form.cleaned_data['bollinger_weight'],
                'stochastic_weight': form.cleaned_data['stochastic_weight'],
                'adx_weight': form.cleaned_data['adx_weight'],
                'ichimoku_weight': form.cleaned_data['ichimoku_weight'],
                'obv_weight': form.cleaned_data['obv_weight']
            }
            profile.save()
            messages.success(request, "Indikator-Gewichtungen erfolgreich gespeichert.")
            return redirect('indicator_weights_settings')
    else:
        form = IndicatorWeightForm(initial=initial_values)

    context = {
        'form': form
    }

    return render(request, 'stock_analyzer/indicator_weights_settings.html', context)


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
