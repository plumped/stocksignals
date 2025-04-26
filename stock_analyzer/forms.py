# stock_analyzer/forms.py
from django import forms
from .models import UserProfile


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = [
            'notify_on_buy_signals', 'notify_on_sell_signals',
            'min_score_for_buy_notification', 'max_score_for_sell_notification',
            'risk_profile'
        ]
        widgets = {
            'min_score_for_buy_notification': forms.NumberInput(attrs={'min': 0, 'max': 100}),
            'max_score_for_sell_notification': forms.NumberInput(attrs={'min': 0, 'max': 100}),
        }


class IndicatorWeightForm(forms.Form):
    """Formular zur Anpassung der Gewichtung der technischen Indikatoren"""
    rsi_weight = forms.IntegerField(min_value=0, max_value=100, initial=10, label="RSI Gewichtung")
    macd_weight = forms.IntegerField(min_value=0, max_value=100, initial=10, label="MACD Gewichtung")
    sma_weight = forms.IntegerField(min_value=0, max_value=100, initial=20, label="Gleitende Durchschnitte Gewichtung")
    bollinger_weight = forms.IntegerField(min_value=0, max_value=100, initial=10, label="Bollinger Bänder Gewichtung")
    stochastic_weight = forms.IntegerField(min_value=0, max_value=100, initial=10, label="Stochastik Gewichtung")
    adx_weight = forms.IntegerField(min_value=0, max_value=100, initial=10, label="ADX Gewichtung")
    ichimoku_weight = forms.IntegerField(min_value=0, max_value=100, initial=10, label="Ichimoku Gewichtung")
    obv_weight = forms.IntegerField(min_value=0, max_value=100, initial=10, label="OBV Gewichtung")

    def clean(self):
        cleaned_data = super().clean()
        # Prüfen, ob die Summe der Gewichtungen 100 ergibt
        weights_sum = sum([
            cleaned_data.get('rsi_weight', 0),
            cleaned_data.get('macd_weight', 0),
            cleaned_data.get('sma_weight', 0),
            cleaned_data.get('bollinger_weight', 0),
            cleaned_data.get('stochastic_weight', 0),
            cleaned_data.get('adx_weight', 0),
            cleaned_data.get('ichimoku_weight', 0),
            cleaned_data.get('obv_weight', 0)
        ])

        if weights_sum != 100:
            raise forms.ValidationError("Die Summe aller Gewichtungen muss 100 ergeben.")

        return cleaned_data