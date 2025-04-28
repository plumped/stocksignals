# stock_analyzer/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('stock/<str:symbol>/', views.stock_detail, name='stock_detail'),
    path('analyze/<str:symbol>/', views.analyze_stock, name='analyze_stock'),
    path('watchlists/', views.my_watchlists, name='my_watchlists'),
    path('watchlists/create/', views.create_watchlist, name='create_watchlist'),
    path('watchlists/<int:watchlist_id>/', views.watchlist_detail, name='watchlist_detail'),
    path('watchlists/<int:watchlist_id>/delete/', views.delete_watchlist, name='delete_watchlist'),

    path('watchlists/add/', views.add_to_watchlist, name='add_to_watchlist'),    # Überprüfe diese URL
    path('watchlists/remove/', views.remove_from_watchlist, name='remove_from_watchlist'),
    path('search/', views.search_stocks, name='search_stocks'),
    path('api/stock/<str:symbol>/', views.api_stock_data, name='api_stock_data'),
    path('batch-analyze/', views.batch_analyze, name='batch_analyze'),
    path('settings/profile/', views.user_profile_settings, name='user_profile_settings'),
    path('stock/<str:symbol>/advanced/', views.advanced_indicators, name='advanced_indicators'),
    path('api/stock/<str:symbol>/advanced/', views.api_advanced_indicators, name='api_advanced_indicators'),

    # Backtesting
    path('backtest/<str:symbol>/', views.run_backtest, name='run_backtest'),

    # Marktanalyse
    path('market/overview/', views.market_overview, name='market_overview'),
    path('market/correlation/', views.correlation_analysis, name='correlation_analysis'),

    # Export-Funktionen
    path('export/stock/<str:symbol>/data/', views.export_stock_data, name='export_stock_data'),
    path('export/stock/<str:symbol>/analysis/', views.export_analysis_results, name='export_analysis_results'),
    # ... andere URLs ...
    path('export_watchlist/<int:watchlist_id>/', views.export_watchlist, name='export_watchlist'),

    path('ml/predict/<str:symbol>/', views.generate_ml_prediction, name='generate_ml_prediction'),
    path('ml/evaluate/<str:symbol>/', views.evaluate_ml_model, name='evaluate_ml_model'),
    path('ml/dashboard/', views.ml_dashboard, name='ml_dashboard'),
    path('ml/batch/', views.batch_ml_predictions_view, name='batch_ml_predictions')


]