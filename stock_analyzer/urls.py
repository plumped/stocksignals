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
    # ... andere URLs ...
    path('export_watchlist/<int:watchlist_id>/', views.export_watchlist, name='export_watchlist'),
]