from django.urls import path
from . import views

app_name = 'guest_system'

urlpatterns = [
    # Main flow
    path('', views.HomeView.as_view(), name='home'),
    path('face-recognition/', views.FaceRecognitionView.as_view(), name='face_recognition'),
    path('gesture-recognition/', views.GestureRecognitionView.as_view(), name='gesture_recognition'),
    path('guest-book/', views.GuestBookView.as_view(), name='guest_book'),
    
    # API endpoints
    path('api/process-face/', views.process_face_recognition, name='process_face'),
    path('api/register-guest/', views.register_guest, name='register_guest'),
    path('api/process-gesture/', views.process_gesture, name='process_gesture'),
    
    # Dashboard
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('api/dashboard-data/', views.dashboard_data, name='dashboard_data'),
    
    # Dashboard API endpoints
    path('api/notifications/<int:notification_id>/read/', views.mark_notification_read, name='mark_notification_read'),
    path('api/notifications/<int:notification_id>/resolve/', views.resolve_notification, name='resolve_notification'),
    path('api/guests/<int:guest_id>/remind/', views.send_reminder, name='send_reminder'),
    path('api/guests/<int:guest_id>/checkout/', views.checkout_guest, name='checkout_guest'),
    
    # Health check
    path('health/', views.health_check, name='health_check'),
]