from django.contrib import admin
from .models import (
    Stock, StockData, AnalysisResult, WatchList, UserProfile,
    MLPrediction, MLModelMetrics,
    Portfolio, Position, Trade
)

class MLPredictionAdmin(admin.ModelAdmin):
    list_display = ('stock', 'date', 'recommendation', 'predicted_return', 'predicted_price', 'confidence')
    list_filter = ('recommendation', 'date')
    search_fields = ('stock__symbol', 'stock__name')
    date_hierarchy = 'date'

class MLModelMetricsAdmin(admin.ModelAdmin):
    list_display = ('stock', 'date', 'accuracy', 'rmse', 'directional_accuracy', 'model_version')
    list_filter = ('date', 'model_version')
    search_fields = ('stock__symbol', 'stock__name')
    date_hierarchy = 'date'

class PortfolioAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'total_value', 'total_gain_loss', 'percent_gain_loss', 'updated_at')
    search_fields = ('name', 'user__username')

class PositionAdmin(admin.ModelAdmin):
    list_display = ('portfolio', 'stock', 'shares', 'average_price', 'current_value', 'gain_loss', 'percent_gain_loss')
    search_fields = ('portfolio__name', 'stock__symbol')

class TradeAdmin(admin.ModelAdmin):
    list_display = ('portfolio', 'stock', 'trade_type', 'date', 'shares', 'price', 'total_value')
    list_filter = ('trade_type', 'date')
    search_fields = ('portfolio__name', 'stock__symbol')
    date_hierarchy = 'date'

# Registriere die Modelle
admin.site.register(Stock)
admin.site.register(StockData)
admin.site.register(AnalysisResult)
admin.site.register(WatchList)
admin.site.register(UserProfile)
admin.site.register(MLPrediction, MLPredictionAdmin)
admin.site.register(MLModelMetrics, MLModelMetricsAdmin)
admin.site.register(Portfolio, PortfolioAdmin)
admin.site.register(Position, PositionAdmin)
admin.site.register(Trade, TradeAdmin)
