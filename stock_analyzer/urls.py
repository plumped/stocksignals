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
    path('ml/get_model_stocks/', views.get_model_stocks, name='get_model_stocks'),
    path('ml/dashboard/', views.ml_dashboard, name='ml_dashboard'),
    path('ml/batch/', views.batch_ml_predictions_view, name='batch_ml_predictions'),

    # Portfolio URLs
    path('portfolio/', views.portfolio_list, name='portfolio_list'),
    path('portfolio/create/', views.portfolio_create, name='portfolio_create'),
    path('portfolio/<int:portfolio_id>/', views.portfolio_detail, name='portfolio_detail'),
    path('portfolio/<int:portfolio_id>/edit/', views.portfolio_edit, name='portfolio_edit'),
    path('portfolio/<int:portfolio_id>/delete/', views.portfolio_delete, name='portfolio_delete'),

    # Position URLs
    path('portfolio/<int:portfolio_id>/positions/', views.position_list, name='position_list'),

    # Trade URLs
    path('portfolio/<int:portfolio_id>/trades/', views.trade_list, name='trade_list'),
    path('portfolio/<int:portfolio_id>/trade/add/', views.trade_add, name='trade_add'),
    path('trade/<int:trade_id>/edit/', views.trade_edit, name='trade_edit'),
    path('trade/<int:trade_id>/delete/', views.trade_delete, name='trade_delete'),

    path('ml/backtest/<str:symbol>/', views.ml_backtest, name='ml_backtest'),
    path('ml/strategy-comparison/<str:symbol>/', views.ml_strategy_comparison, name='ml_strategy_comparison'),
    path('ml/batch-backtest/', views.ml_batch_backtest, name='ml_batch_backtest'),
    path('api/ml/backtest/<str:symbol>/', views.api_ml_backtest, name='api_ml_backtest'),
    path('api/ml_metrics/<str:symbol>/', views.api_ml_metrics, name='api_ml_metrics'),

    # Portfolio Performance
    path('portfolio/<int:portfolio_id>/performance/', views.portfolio_performance, name='portfolio_performance'),

]
