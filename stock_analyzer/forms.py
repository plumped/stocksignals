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