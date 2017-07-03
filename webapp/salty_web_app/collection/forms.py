from django.forms import ModelForm

from collection.models import StatsModels

class StatsModelsForm(ModelForm):
    class Meta: 
        model = StatsModels
        fields = ('name', 'description',)
