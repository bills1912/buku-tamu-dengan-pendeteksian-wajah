from django.contrib import admin
from .models import Guest, VisitLog

@admin.register(Guest)
class GuestAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'email']

@admin.register(VisitLog)
class VisitLogAdmin(admin.ModelAdmin):
    list_display = ['guest', 'visit_purpose', 'visit_date']
    list_filter = ['visit_purpose', 'visit_date']
    search_fields = ['guest__name']
