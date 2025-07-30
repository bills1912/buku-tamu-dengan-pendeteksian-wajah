from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.utils import timezone
from django.db.models import Count, Q
from datetime import datetime, timedelta
import json
import base64
import numpy as np
from .models import Guest, VisitLog, Notification

# LAZY LOADING SERVICES - Avoid immediate import to prevent pygame startup message
face_service = None
kinetic_service = None
audio_service = None

def get_face_service():
    """Lazy initialization of face recognition service"""
    global face_service
    if face_service is None:
        try:
            from .services.face_recognition_service import FaceRecognitionService
            face_service = FaceRecognitionService()
        except Exception as e:
            print(f"⚠️ Face recognition service failed to load: {e}")
            face_service = None
    return face_service

def get_kinetic_service():
    """Lazy initialization of kinetic recognition service"""
    global kinetic_service
    if kinetic_service is None:
        try:
            from .services.kinetic_service import KineticRecognitionService
            kinetic_service = KineticRecognitionService()
        except Exception as e:
            print(f"⚠️ Kinetic recognition service failed to load: {e}")
            kinetic_service = None
    return kinetic_service

def get_audio_service():
    """Lazy initialization of audio service with single-message support"""
    global audio_service
    if audio_service is None:
        try:
            try:
                from .services.audio_service import get_audio_service
                audio_service = get_audio_service()
            except:
                from .services.audio_service import AudioService
                audio_service = AudioService()
        except Exception as e:
            print(f"⚠️ Audio service failed to load: {e}")
            audio_service = None
    return audio_service

def create_notification(guest, notification_type, title, message):
    """Helper function to create notifications"""
    try:
        Notification.objects.create(
            guest=guest,
            notification_type=notification_type,
            title=title,
            message=message
        )
    except Exception as e:
        print(f"Error creating notification: {e}")

class HomeView(View):
    def get(self, request):
        return render(request, 'guest_system/home.html')

class FaceRecognitionView(View):
    def get(self, request):
        return render(request, 'guest_system/face_recognition.html')

@csrf_exempt
def process_face_recognition(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            
            if not image_data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No image data provided'
                })
            
            # Lazy import cv2 to avoid startup issues
            try:
                import cv2
            except ImportError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'OpenCV not available'
                })
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data.split(',')[1])
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Invalid image data'
                    })
            except Exception as decode_error:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Image decode error: {decode_error}'
                })
            
            # Get face service
            face_svc = get_face_service()
            if not face_svc:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Face recognition service not available'
                })
            
            # Process face recognition
            try:
                face_locations, face_encodings = face_svc.process_frame(frame)
                
                if face_encodings:
                    face_encoding = face_encodings[0]
                    guest_id, guest_name = face_svc.recognize_face(face_encoding)
                    
                    # Get audio service
                    audio_svc = get_audio_service()
                    
                    if guest_id:
                        # RETURNING GUEST - Use single combined message
                        if audio_svc:
                            if hasattr(audio_svc, 'welcome_and_instruct_returning_guest'):
                                audio_svc.welcome_and_instruct_returning_guest(guest_name)
                            else:
                                combined_message = f"Selamat datang kembali, {guest_name}. Silakan tunjukkan gestur: angka 1 untuk pelayanan data, angka 2 untuk menemui pegawai."
                                if hasattr(audio_svc, 'speak'):
                                    audio_svc.speak(combined_message)
                                else:
                                    audio_svc.welcome_guest(guest_name)
                        
                        return JsonResponse({
                            'status': 'returning_guest',
                            'guest_id': guest_id,
                            'guest_name': guest_name
                        })
                    else:
                        # NEW GUEST - Use single message
                        if audio_svc:
                            if hasattr(audio_svc, 'welcome_and_instruct_new_guest'):
                                audio_svc.welcome_and_instruct_new_guest()
                            else:
                                if hasattr(audio_svc, 'speak'):
                                    audio_svc.speak("Selamat datang! Anda adalah tamu baru. Silakan isi data diri Anda terlebih dahulu.")
                                else:
                                    audio_svc.welcome_new_guest()
                        
                        return JsonResponse({
                            'status': 'new_guest',
                            'face_encoding': face_encoding.tolist(),
                            'image_data': image_data
                        })
                else:
                    return JsonResponse({
                        'status': 'no_face_detected',
                        'message': 'Tidak ada wajah yang terdeteksi'
                    })
                    
            except Exception as face_error:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Face recognition error: {face_error}'
                })
                
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON data'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({'status': 'invalid_request'})

@csrf_exempt
def register_guest(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Validate required fields
            if not data.get('name'):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Nama wajib diisi'
                })
            
            # Create new guest
            guest = Guest.objects.create(
                name=data['name'],
                email=data.get('email', ''),
                phone=data.get('phone', ''),
                company=data.get('company', '')
            )
            
            # Save face encoding
            if data.get('face_encoding'):
                try:
                    face_encoding = np.array(data['face_encoding'])
                    guest.set_face_encoding(face_encoding)
                except Exception as encoding_error:
                    print(f"Face encoding error: {encoding_error}")
            
            # Save face image
            try:
                face_svc = get_face_service()
                if face_svc and data.get('image_data'):
                    image_path = face_svc.save_face_image(data['image_data'], guest.name)
                    if image_path:
                        guest.face_image = image_path
            except Exception as image_error:
                print(f"Face image save error: {image_error}")
            
            guest.save()
            
            # Create notification for new registration
            create_notification(
                guest=guest,
                notification_type='new_registration',
                title='Tamu Baru Terdaftar',
                message=f'Tamu baru {guest.name} telah berhasil mendaftar di sistem.'
            )
            
            # Reload known faces
            try:
                face_svc = get_face_service()
                if face_svc:
                    face_svc.load_known_faces()
            except Exception as reload_error:
                print(f"Face service reload error: {reload_error}")
            
            # Give registration success audio
            audio_svc = get_audio_service()
            if audio_svc:
                if hasattr(audio_svc, 'registration_success'):
                    audio_svc.registration_success()
                elif hasattr(audio_svc, 'speak'):
                    audio_svc.speak("Pendaftaran berhasil. Sekarang silakan tunjukkan gestur tangan untuk memilih layanan.")
            
            return JsonResponse({
                'status': 'success',
                'guest_id': guest.id,
                'message': 'Pendaftaran berhasil'
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({'status': 'invalid_request'})

class GestureRecognitionView(View):
    def get(self, request):
        guest_id = request.GET.get('guest_id')
        guest = None
        if guest_id:
            guest = get_object_or_404(Guest, id=guest_id)
        
        return render(request, 'guest_system/gesture_recognition.html', {
            'guest': guest
        })

@csrf_exempt
def process_gesture(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            
            if not image_data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No image data provided'
                })
            
            # Lazy import cv2
            try:
                import cv2
            except ImportError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'OpenCV not available'
                })
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data.split(',')[1])
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Invalid image data'
                    })
            except Exception as decode_error:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Image decode error: {decode_error}'
                })
            
            # Get kinetic service
            kinetic_svc = get_kinetic_service()
            if not kinetic_svc:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Gesture recognition not available'
                })
            
            # Detect gesture
            try:
                gesture, hand_landmarks = kinetic_svc.detect_gesture(frame)
                
                if gesture:
                    audio_svc = get_audio_service()
                    
                    if gesture == 1:
                        if audio_svc:
                            if hasattr(audio_svc, 'direct_to_pst'):
                                audio_svc.direct_to_pst()
                            elif hasattr(audio_svc, 'speak'):
                                audio_svc.speak("Terima kasih. Silakan menuju ruang PST untuk pelayanan data.")
                        
                        return JsonResponse({
                            'status': 'success',
                            'gesture': 1,
                            'action': 'pst',
                            'message': 'Menuju ruang PST'
                        })
                    elif gesture == 2:
                        if audio_svc:
                            if hasattr(audio_svc, 'direct_to_staff'):
                                audio_svc.direct_to_staff()
                            elif hasattr(audio_svc, 'speak'):
                                audio_svc.speak("Terima kasih. Silakan langsung menemui pegawai atau kepala kantor.")
                        
                        return JsonResponse({
                            'status': 'success',
                            'gesture': 2,
                            'action': 'staff',
                            'message': 'Menuju pegawai/kepala kantor'
                        })
                
                return JsonResponse({
                    'status': 'no_gesture',
                    'message': 'Gestur tidak terdeteksi'
                })
                
            except Exception as gesture_error:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Gesture recognition error: {gesture_error}'
                })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON data'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({'status': 'invalid_request'})

class GuestBookView(View):
    def get(self, request):
        guest_id = request.GET.get('guest_id')
        purpose = request.GET.get('purpose')
        
        guest = None
        if guest_id:
            guest = get_object_or_404(Guest, id=guest_id)
        
        return render(request, 'guest_system/guest_book.html', {
            'guest': guest,
            'purpose': purpose
        })
    
    def post(self, request):
        try:
            guest_id = request.POST.get('guest_id')
            purpose = request.POST.get('purpose')
            description = request.POST.get('description')
            urgency = request.POST.get('urgency', 'low')
            expected_duration = request.POST.get('expected_duration')
            
            guest = get_object_or_404(Guest, id=guest_id)
            
            # Create visit log
            visit_log = VisitLog.objects.create(
                guest=guest,
                visit_purpose=purpose,
                visit_description=description,
                urgency_level=urgency,
                expected_duration=int(expected_duration) if expected_duration else None
            )
            
            # Create notification for completed guestbook
            create_notification(
                guest=guest,
                notification_type='guestbook_completed',
                title='Buku Tamu Telah Diisi',
                message=f'{guest.name} telah mengisi buku tamu dengan tujuan: {dict(Guest.VISIT_PURPOSE_CHOICES)[purpose]}'
            )
            
            # Give completion audio with direction
            audio_svc = get_audio_service()
            if audio_svc:
                if hasattr(audio_svc, 'guestbook_complete'):
                    audio_svc.guestbook_complete(purpose)
                elif hasattr(audio_svc, 'speak'):
                    if purpose == 'data_service':
                        audio_svc.speak("Buku tamu telah diisi. Silakan menuju ruang PST untuk pelayanan data.")
                    else:
                        audio_svc.speak("Buku tamu telah diisi. Silakan menemui pegawai atau kepala kantor.")
            
            return JsonResponse({
                'status': 'success',
                'message': 'Buku tamu berhasil diisi'
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })

# Dashboard Views
class DashboardView(View):
    def get(self, request):
        return render(request, 'guest_system/dashboard.html')

def dashboard_data(request):
    """API endpoint to get dashboard data"""
    try:
        now = timezone.now()
        today = now.date()
        
        # Statistics
        total_guests = Guest.objects.count()
        today_guests = Guest.objects.filter(created_at__date=today).count()
        active_visits = VisitLog.objects.filter(check_out_time__isnull=True).count()
        
        # Guests who haven't filled guestbook (have no visit logs)
        pending_guestbook = Guest.objects.filter(visitlog__isnull=True).count()
        
        # Unread notifications
        unread_notifications = Notification.objects.filter(is_read=False).count()
        
        # Recent notifications
        notifications = []
        for notification in Notification.objects.filter(is_resolved=False)[:10]:
            notifications.append({
                'id': notification.id,
                'type': notification.notification_type,
                'title': notification.title,
                'message': notification.message,
                'guest_name': notification.guest.name,
                'time_ago': notification.time_ago,
                'is_read': notification.is_read
            })
        
        # Recent guests
        recent_guests = []
        for guest in Guest.objects.order_by('-created_at')[:10]:
            status = guest.status
            status_info = {
                'registered': {'text': 'Terdaftar', 'class': 'registered'},
                'pending_guestbook': {'text': 'Belum Isi Buku Tamu', 'class': 'pending'},
                'active_visit': {'text': 'Sedang Berkunjung', 'class': 'completed'},
                'checked_out': {'text': 'Sudah Check-out', 'class': 'checked-out'}
            }
            
            current_visit = guest.current_visit
            
            recent_guests.append({
                'id': guest.id,
                'name': guest.name,
                'email': guest.email,
                'company': guest.company,
                'status': status,
                'status_info': status_info.get(status, {'text': 'Unknown', 'class': 'unknown'}),
                'registration_time': guest.created_at.strftime('%H:%M'),
                'face_image_url': guest.face_image.url if guest.face_image else None,
                'current_visit': {
                    'purpose': dict(Guest.VISIT_PURPOSE_CHOICES)[current_visit.visit_purpose],
                    'description': current_visit.visit_description,
                    'duration': f"{current_visit.duration_minutes} menit"
                } if current_visit else None
            })
        
        # Active visits
        active_visits_data = []
        for visit in VisitLog.objects.filter(check_out_time__isnull=True):
            active_visits_data.append({
                'id': visit.id,
                'guest_name': visit.guest.name,
                'purpose': dict(Guest.VISIT_PURPOSE_CHOICES)[visit.visit_purpose],
                'duration_minutes': visit.duration_minutes,
                'check_in_time': visit.check_in_time.strftime('%H:%M'),
                'urgency_level': visit.urgency_level,
                'is_overdue': visit.is_overdue
            })
        
        # Daily statistics (last 7 days)
        daily_stats = []
        for i in range(7):
            date = today - timedelta(days=6-i)
            registrations = Guest.objects.filter(created_at__date=date).count()
            daily_stats.append({
                'day': date.strftime('%m/%d'),
                'registrations': registrations
            })
        
        # Purpose breakdown (today)
        purpose_breakdown = []
        for purpose_code, purpose_name in Guest.VISIT_PURPOSE_CHOICES:
            count = VisitLog.objects.filter(
                visit_date__date=today,
                visit_purpose=purpose_code
            ).count()
            if count > 0:
                purpose_breakdown.append({
                    'purpose': purpose_code,
                    'purpose_display': purpose_name,
                    'count': count
                })
        
        return JsonResponse({
            'status': 'success',
            'data': {
                'statistics': {
                    'total_guests': total_guests,
                    'today_guests': today_guests,
                    'active_visits': active_visits,
                    'pending_guestbook': pending_guestbook,
                    'unread_notifications': unread_notifications
                },
                'notifications': notifications,
                'recent_guests': recent_guests,
                'active_visits': active_visits_data,
                'daily_stats': daily_stats,
                'purpose_breakdown': purpose_breakdown
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

# Dashboard API endpoints
@csrf_exempt
def mark_notification_read(request, notification_id):
    """Mark notification as read"""
    if request.method == 'POST':
        try:
            notification = get_object_or_404(Notification, id=notification_id)
            notification.is_read = True
            notification.save()
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'invalid_request'})

@csrf_exempt
def resolve_notification(request, notification_id):
    """Resolve notification"""
    if request.method == 'POST':
        try:
            notification = get_object_or_404(Notification, id=notification_id)
            notification.is_resolved = True
            notification.is_read = True
            notification.save()
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'invalid_request'})

@csrf_exempt
def send_reminder(request, guest_id):
    """Send reminder to guest"""
    if request.method == 'POST':
        try:
            guest = get_object_or_404(Guest, id=guest_id)
            
            # Create reminder notification
            create_notification(
                guest=guest,
                notification_type='guestbook_reminder',
                title='Pengingat Isi Buku Tamu',
                message=f'Pengingat telah dikirim kepada {guest.name} untuk mengisi buku tamu.'
            )
            
            # Here you could add actual reminder logic (SMS, email, etc.)
            
            return JsonResponse({
                'status': 'success',
                'message': 'Reminder sent successfully'
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'invalid_request'})

@csrf_exempt
def checkout_guest(request, guest_id):
    """Check out guest"""
    if request.method == 'POST':
        try:
            guest = get_object_or_404(Guest, id=guest_id)
            
            # Find active visit and check out
            active_visit = VisitLog.objects.filter(
                guest=guest,
                check_out_time__isnull=True
            ).first()
            
            if active_visit:
                active_visit.check_out_time = timezone.now()
                active_visit.save()
                
                # Create checkout notification
                create_notification(
                    guest=guest,
                    notification_type='guest_checkout',
                    title='Tamu Check-out',
                    message=f'{guest.name} telah check-out setelah {active_visit.duration_minutes} menit berkunjung.'
                )
            
            return JsonResponse({
                'status': 'success',
                'message': 'Guest checked out successfully'
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'invalid_request'})

# Background task to create notifications for guests who haven't filled guestbook
def check_pending_guestbook():
    """Background task to check for guests who haven't filled guestbook"""
    from django.utils import timezone
    from datetime import timedelta
    
    # Find guests registered more than 10 minutes ago but haven't filled guestbook
    cutoff_time = timezone.now() - timedelta(minutes=10)
    pending_guests = Guest.objects.filter(
        created_at__lt=cutoff_time,
        visitlog__isnull=True
    ).exclude(
        notification__notification_type='guestbook_reminder',
        notification__created_at__gt=timezone.now() - timedelta(hours=1)
    )
    
    for guest in pending_guests:
        create_notification(
            guest=guest,
            notification_type='guestbook_reminder',
            title='Belum Mengisi Buku Tamu',
            message=f'{guest.name} telah terdaftar namun belum mengisi buku tamu selama {int((timezone.now() - guest.created_at).total_seconds() / 60)} menit.'
        )

# Health check view for monitoring
def health_check(request):
    """Simple health check endpoint"""
    try:
        guest_count = Guest.objects.count()
        visit_count = VisitLog.objects.count()
        notification_count = Notification.objects.count()
        
        # Check service availability
        services_status = {
            'face_recognition': get_face_service() is not None,
            'kinetic_recognition': get_kinetic_service() is not None,
            'audio': get_audio_service() is not None,
        }
        
        return JsonResponse({
            'status': 'healthy',
            'database': 'ok',
            'guests': guest_count,
            'visits': visit_count,
            'notifications': notification_count,
            'services': services_status
        })
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)