from django.contrib import admin
from .models import Stock, StockData, AnalysisResult, WatchList, UserProfile, MLPrediction, MLModelMetrics

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

# Registriere die Modelle
admin.site.register(Stock)
admin.site.register(StockData)
admin.site.register(AnalysisResult)
admin.site.register(WatchList)
admin.site.register(UserProfile)
admin.site.register(MLPrediction, MLPredictionAdmin)
admin.site.register(MLModelMetrics, MLModelMetricsAdmin)