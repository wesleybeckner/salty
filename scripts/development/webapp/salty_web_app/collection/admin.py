from django.contrib import admin
#import your model
from collection.models import StatsModels

#set up automated slug creation
class StatsModelsAdmin(admin.ModelAdmin):
    model = StatsModels
    list_display = ('name', 'description',)
    prepolulated_fields = {'slug': ('name',)}

# Register your models here.
admin.site.register(StatsModels, StatsModelsAdmin)
