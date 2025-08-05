from django.urls import path
from . import views

app_name = 'guest_system'

urlpatterns = [
    # Main flow
    path('', views.HomeView.as_view(), name='home'),
    path('face-recognition/', views.FaceRecognitionView.as_view(), name='face_recognition'),
    path('gesture-recognition/', views.GestureRecognitionView.as_view(), name='gesture_recognition'),
    path('staff-selection/', views.StaffSelectionView.as_view(), name='staff_selection'),
    path('guest-book/', views.GuestBookView.as_view(), name='guest_book'),
    
    # API endpoints
    path('api/process-face/', views.process_face_recognition, name='process_face'),
    path('api/register-guest/', views.register_guest, name='register_guest'),
    path('api/process-gesture/', views.process_gesture, name='process_gesture'),
    path('api/get-staff-list/', views.get_staff_list, name='get_staff_list'),
    
    # Dashboard
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('api/dashboard-data/', views.dashboard_data, name='dashboard_data'),
    
    # Dashboard API endpoints
    path('api/notifications/<int:notification_id>/read/', views.mark_notification_read, name='mark_notification_read'),
    path('api/notifications/<int:notification_id>/resolve/', views.resolve_notification, name='resolve_notification'),
    path('api/guests/<int:guest_id>/remind/', views.send_reminder, name='send_reminder'),
    path('api/guests/<int:guest_id>/checkout/', views.checkout_guest, name='checkout_guest'),
    
    # Version & Changelog
    path('changelog/', views.ChangelogView.as_view(), name='changelog'),
    path('api/changelog-data/', views.changelog_data, name='changelog_data'),
    
    # Staff management
    path('api/staff/<int:staff_id>/status/', views.update_staff_status, name='update_staff_status'),
    path('api/staff/<int:staff_id>/test-whatsapp/', views.test_staff_whatsapp, name='test_staff_whatsapp'),
    path('api/staff/create/', views.create_staff, name='create_staff'),
    path('api/staff/<int:staff_id>/update/', views.update_staff, name='update_staff'),
    path('api/staff/<int:staff_id>/delete/', views.delete_staff, name='delete_staff'),
    path('api/staff/<int:staff_id>/detail/', views.get_staff_detail, name='get_staff_detail'),
    path('api/staff/bulk-update-status/', views.bulk_update_staff_status, name='bulk_update_staff_status'),
    
    # Health check
    path('health/', views.health_check, name='health_check'),
]