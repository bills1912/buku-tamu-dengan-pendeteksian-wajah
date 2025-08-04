from django.contrib import admin
from .models import Guest, VisitLog, Notification, Version, ChangelogItem, Staff

@admin.register(Guest)
class GuestAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'email']
    readonly_fields = ['face_encoding']

@admin.register(VisitLog)
class VisitLogAdmin(admin.ModelAdmin):
    list_display = ['guest', 'visit_purpose', 'staff_to_meet', 'visit_date', 'whatsapp_sent']
    list_filter = ['visit_purpose', 'visit_date', 'whatsapp_sent', 'urgency_level']
    search_fields = ['guest__name', 'staff_to_meet__name']
    raw_id_fields = ['guest', 'staff_to_meet']

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['title', 'guest', 'related_staff', 'notification_type', 'is_read', 'created_at']
    list_filter = ['notification_type', 'is_read', 'is_resolved', 'created_at']
    search_fields = ['title', 'guest__name', 'related_staff__name']
    raw_id_fields = ['guest', 'related_staff']

class ChangelogItemInline(admin.TabularInline):
    model = ChangelogItem
    extra = 1
    fields = ['item_type', 'title', 'description', 'is_highlighted']

@admin.register(Version)
class VersionAdmin(admin.ModelAdmin):
    list_display = ['version_number', 'title', 'release_date', 'is_current']
    list_filter = ['is_current', 'release_date']
    search_fields = ['version_number', 'title']
    inlines = [ChangelogItemInline]
    readonly_fields = ['created_at']
    
    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        if obj.is_current:
            # Ensure only one version is marked as current
            Version.objects.exclude(pk=obj.pk).update(is_current=False)

@admin.register(ChangelogItem)
class ChangelogItemAdmin(admin.ModelAdmin):
    list_display = ['version', 'item_type', 'title', 'is_highlighted']
    list_filter = ['item_type', 'is_highlighted', 'version']
    search_fields = ['title', 'description']
    raw_id_fields = ['version']

@admin.register(Staff)
class StaffAdmin(admin.ModelAdmin):
    list_display = ['name', 'position', 'department', 'current_status', 'whatsapp_enabled', 'is_active']
    list_filter = ['department', 'current_status', 'whatsapp_enabled', 'is_active']
    search_fields = ['name', 'position', 'phone_number', 'email']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Informasi Dasar', {
            'fields': ('name', 'position', 'department', 'photo')
        }),
        ('Kontak', {
            'fields': ('phone_number', 'email', 'office_room')
        }),
        ('Status', {
            'fields': ('current_status', 'status_message', 'is_active')
        }),
        ('Notifikasi', {
            'fields': ('whatsapp_enabled',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['mark_available', 'mark_busy', 'enable_whatsapp', 'disable_whatsapp']
    
    def mark_available(self, request, queryset):
        queryset.update(current_status='available')
        self.message_user(request, f"{queryset.count()} staff marked as available.")
    mark_available.short_description = "Mark selected staff as available"
    
    def mark_busy(self, request, queryset):
        queryset.update(current_status='busy')
        self.message_user(request, f"{queryset.count()} staff marked as busy.")
    mark_busy.short_description = "Mark selected staff as busy"
    
    def enable_whatsapp(self, request, queryset):
        queryset.update(whatsapp_enabled=True)
        self.message_user(request, f"WhatsApp enabled for {queryset.count()} staff.")
    enable_whatsapp.short_description = "Enable WhatsApp notifications"
    
    def disable_whatsapp(self, request, queryset):
        queryset.update(whatsapp_enabled=False)
        self.message_user(request, f"WhatsApp disabled for {queryset.count()} staff.")
    disable_whatsapp.short_description = "Disable WhatsApp notifications"