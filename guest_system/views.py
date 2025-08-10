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
from .models import Guest, VisitLog, Notification, Version, ChangelogItem, Staff
from .services.whatsapp_service import whatsapp_service
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.utils import timezone
from django.db.models import Q
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
import re
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.utils import timezone
from django.shortcuts import get_object_or_404
import json

from .models import Version, ChangelogItem
logger = logging.getLogger(__name__)

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

def create_notification(guest, notification_type, title, message, related_staff=None):
    """Helper function to create notifications"""
    try:
        Notification.objects.create(
            guest=guest,
            notification_type=notification_type,
            title=title,
            message=message,
            related_staff=related_staff
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
                                audio_svc.speak("Terima kasih. Silakan pilih pegawai yang ingin Anda temui.")
                        
                        return JsonResponse({
                            'status': 'success',
                            'gesture': 2,
                            'action': 'staff_selection',
                            'message': 'Pilih staff yang ingin ditemui'
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

class StaffSelectionView(View):
    def get(self, request):
        guest_id = request.GET.get('guest_id')
        guest = None
        if guest_id:
            guest = get_object_or_404(Guest, id=guest_id)
        
        return render(request, 'guest_system/staff_selection.html', {
            'guest': guest
        })

def get_staff_list(request):
    """API endpoint to get list of available staff"""
    try:
        staff_list = []
        for staff in Staff.objects.filter(is_active=True).order_by('department', 'name'):
            staff_data = {
                'id': staff.id,
                'name': staff.name,
                'position': staff.position,
                'department': staff.department,
                'department_display': staff.department_display,
                'current_status': staff.current_status,
                'status_message': staff.status_message,
                'office_room': staff.office_room,
                'photo': staff.photo.url if staff.photo else None,
                'whatsapp_enabled': staff.whatsapp_enabled,
            }
            staff_list.append(staff_data)
        
        return JsonResponse({
            'status': 'success',
            'data': staff_list
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

class GuestBookView(View):
    def get(self, request):
        guest_id = request.GET.get('guest_id')
        purpose = request.GET.get('purpose')
        staff_id = request.GET.get('staff_id')
        
        guest = None
        staff = None
        
        if guest_id:
            guest = get_object_or_404(Guest, id=guest_id)
            
        if staff_id:
            staff = get_object_or_404(Staff, id=staff_id)
        
        return render(request, 'guest_system/guest_book.html', {
            'guest': guest,
            'purpose': purpose,
            'staff': staff
        })
    
    def post(self, request):
        try:
            guest_id = request.POST.get('guest_id')
            purpose = request.POST.get('purpose')
            description = request.POST.get('description')
            urgency = request.POST.get('urgency', 'low')
            expected_duration = request.POST.get('expected_duration')
            staff_id = request.POST.get('staff_id')
            
            guest = get_object_or_404(Guest, id=guest_id)
            staff = None
            
            if staff_id:
                staff = get_object_or_404(Staff, id=staff_id)
            
            # Create visit log
            visit_log = VisitLog.objects.create(
                guest=guest,
                visit_purpose=purpose,
                visit_description=description,
                urgency_level=urgency,
                expected_duration=int(expected_duration) if expected_duration else None,
                staff_to_meet=staff
            )
            
            # Create notification for completed guestbook
            create_notification(
                guest=guest,
                notification_type='guestbook_completed',
                title='Buku Tamu Telah Diisi',
                message=f'{guest.name} telah mengisi buku tamu dengan tujuan: {dict(Guest.VISIT_PURPOSE_CHOICES)[purpose]}',
                related_staff=staff
            )
            
            # Send WhatsApp notification if staff is selected
            if staff and staff.whatsapp_enabled:
                try:
                    whatsapp_result = whatsapp_service.send_guest_meeting_notification(
                        staff=staff,
                        guest=guest,
                        visit_log=visit_log
                    )
                    
                    if whatsapp_result['success']:
                        visit_log.whatsapp_sent = True
                        visit_log.whatsapp_sent_at = timezone.now()
                        visit_log.save()
                        
                        # Create WhatsApp success notification
                        create_notification(
                            guest=guest,
                            notification_type='whatsapp_sent',
                            title='WhatsApp Terkirim',
                            message=f'Notifikasi WhatsApp berhasil dikirim ke {staff.name}',
                            related_staff=staff
                        )
                    else:
                        # Create WhatsApp failure notification
                        create_notification(
                            guest=guest,
                            notification_type='staff_meeting_request',
                            title='Permintaan Bertemu Staff',
                            message=f'{guest.name} ingin bertemu dengan {staff.name}. WhatsApp gagal dikirim: {whatsapp_result["message"]}',
                            related_staff=staff
                        )
                        
                except Exception as whatsapp_error:
                    print(f"WhatsApp error: {whatsapp_error}")
                    create_notification(
                        guest=guest,
                        notification_type='staff_meeting_request',
                        title='Permintaan Bertemu Staff',
                        message=f'{guest.name} ingin bertemu dengan {staff.name}. Error WhatsApp: {str(whatsapp_error)}',
                        related_staff=staff
                    )
            
            # Give completion audio with direction
            audio_svc = get_audio_service()
            if audio_svc:
                if hasattr(audio_svc, 'guestbook_complete'):
                    audio_svc.guestbook_complete(purpose)
                elif hasattr(audio_svc, 'speak'):
                    if purpose == 'data_service':
                        audio_svc.speak("Buku tamu telah diisi. Silakan menuju ruang PST untuk pelayanan data.")
                    elif purpose == 'meet_staff' and staff:
                        audio_svc.speak(f"Buku tamu telah diisi. Silakan menuju {staff.office_room or 'ruang kerja'} untuk menemui {staff.name}.")
                    else:
                        audio_svc.speak("Buku tamu telah diisi. Terima kasih atas kunjungan Anda.")
            
            return JsonResponse({
                'status': 'success',
                'message': 'Buku tamu berhasil diisi',
                'whatsapp_sent': visit_log.whatsapp_sent
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
                'staff_name': notification.related_staff.name if notification.related_staff else None,
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
                    'duration': f"{current_visit.duration_minutes} menit",
                    'staff_name': current_visit.staff_to_meet.name if current_visit.staff_to_meet else None
                } if current_visit else None
            })
        
        # Active visits
        active_visits_data = []
        for visit in VisitLog.objects.filter(check_out_time__isnull=True):
            active_visits_data.append({
                'id': visit.id,
                'guest_name': visit.guest.name,
                'purpose': dict(Guest.VISIT_PURPOSE_CHOICES)[visit.visit_purpose],
                'staff_name': visit.staff_to_meet.name if visit.staff_to_meet else None,
                'duration_minutes': visit.duration_minutes,
                'check_in_time': visit.check_in_time.strftime('%H:%M'),
                'urgency_level': visit.urgency_level,
                'is_overdue': visit.is_overdue,
                'whatsapp_sent': visit.whatsapp_sent
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

# Changelog Views
class ChangelogView(View):
    def get(self, request):
        return render(request, 'guest_system/changelog.html')

def changelog_data(request):
    """API endpoint to get changelog data"""
    try:
        # Get current version
        current_version = Version.objects.filter(is_current=True).first()
        
        # Get all versions with their changelog items
        versions = []
        for version in Version.objects.all():
            changelog_items = []
            for item in version.changelog_items.all():
                changelog_items.append({
                    'item_type': item.item_type,
                    'title': item.title,
                    'description': item.description,
                    'is_highlighted': item.is_highlighted,
                })
            
            versions.append({
                'version_number': version.version_number,
                'title': version.title,
                'description': version.description,
                'release_date': version.release_date.isoformat(),
                'is_current': version.is_current,
                'changelog_items': changelog_items
            })
        
        return JsonResponse({
            'status': 'success',
            'data': {
                'current_version': {
                    'version_number': current_version.version_number,
                    'title': current_version.title
                } if current_version else None,
                'versions': versions
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
                    message=f'{guest.name} telah check-out setelah {active_visit.duration_minutes} menit berkunjung.',
                    related_staff=active_visit.staff_to_meet
                )
            
            return JsonResponse({
                'status': 'success',
                'message': 'Guest checked out successfully'
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'invalid_request'})

# Staff management endpoints
@csrf_exempt
def update_staff_status(request, staff_id):
    """Update staff status"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            staff = get_object_or_404(Staff, id=staff_id)
            
            staff.current_status = data.get('status', staff.current_status)
            staff.status_message = data.get('message', staff.status_message)
            staff.save()
            
            return JsonResponse({
                'status': 'success',
                'message': f'Status {staff.name} berhasil diperbarui'
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'invalid_request'})

@csrf_exempt
def test_staff_whatsapp(request, staff_id):
    """Test WhatsApp notification to staff"""
    if request.method == 'POST':
        try:
            staff = get_object_or_404(Staff, id=staff_id)
            
            result = whatsapp_service.send_test_message(
                phone_number=staff.phone_number,
                staff_name=staff.name
            )
            
            return JsonResponse({
                'status': 'success' if result['success'] else 'error',
                'message': result['message']
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
        staff_count = Staff.objects.filter(is_active=True).count()
        
        # Check service availability
        services_status = {
            'face_recognition': get_face_service() is not None,
            'kinetic_recognition': get_kinetic_service() is not None,
            'audio': get_audio_service() is not None,
            'whatsapp': whatsapp_service.enabled,
        }
        
        return JsonResponse({
            'status': 'healthy',
            'database': 'ok',
            'guests': guest_count,
            'visits': visit_count,
            'notifications': notification_count,
            'staff': staff_count,
            'services': services_status
        })
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)
        
@csrf_exempt
@require_http_methods(["POST"])
def create_staff(request):
    """Create new staff member"""
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['name', 'position', 'department', 'phone_number']
        for field in required_fields:
            if not data.get(field):
                return JsonResponse({
                    'status': 'error',
                    'message': f'Field {field} wajib diisi'
                })
        
        # Validate phone number format
        phone_number = data['phone_number'].strip()
        if not re.match(r'^(\+62|62|0)[0-9]{8,12}$', phone_number):
            return JsonResponse({
                'status': 'error',
                'message': 'Format nomor telepon tidak valid. Gunakan format: 08xxxx, 62xxxx, atau +62xxxx'
            })
        
        # Validate email if provided
        email = data.get('email', '').strip()
        if email:
            try:
                validate_email(email)
            except ValidationError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Format email tidak valid'
                })
        
        # Validate department
        valid_departments = [choice[0] for choice in Staff.DEPARTMENTS]
        if data['department'] not in valid_departments:
            return JsonResponse({
                'status': 'error',
                'message': 'Departemen tidak valid'
            })
        
        # Validate status
        valid_statuses = [choice[0] for choice in Staff.STATUS_CHOICES]
        current_status = data.get('current_status', 'available')
        if current_status not in valid_statuses:
            return JsonResponse({
                'status': 'error',
                'message': 'Status tidak valid'
            })
        
        # Check if phone number already exists
        if Staff.objects.filter(phone_number=phone_number).exists():
            return JsonResponse({
                'status': 'error',
                'message': 'Nomor telepon sudah terdaftar untuk staff lain'
            })
        
        # Check if email already exists (if provided)
        if email and Staff.objects.filter(email=email).exists():
            return JsonResponse({
                'status': 'error',
                'message': 'Email sudah terdaftar untuk staff lain'
            })
        
        # Create staff
        staff = Staff.objects.create(
            name=data['name'].strip(),
            position=data['position'].strip(),
            department=data['department'],
            phone_number=phone_number,
            email=email if email else None,
            office_room=data.get('office_room', '').strip() or None,
            current_status=current_status,
            status_message=data.get('status_message', '').strip() or None,
            whatsapp_enabled=data.get('whatsapp_enabled', True),
            is_active=data.get('is_active', True)
        )
        
        # Create notification for new staff
        try:
            create_notification(
                guest=None,  # System notification
                notification_type='staff_added',
                title='Staff Baru Ditambahkan',
                message=f'Staff baru {staff.name} telah ditambahkan ke sistem dengan jabatan {staff.position}'
            )
        except:
            pass  # Don't fail if notification creation fails
        
        logger.info(f"New staff created: {staff.name} (ID: {staff.id})")
        
        return JsonResponse({
            'status': 'success',
            'message': 'Staff berhasil ditambahkan',
            'staff_id': staff.id,
            'data': {
                'id': staff.id,
                'name': staff.name,
                'position': staff.position,
                'department': staff.department,
                'department_display': staff.department_display,
                'phone_number': staff.phone_number,
                'email': staff.email,
                'office_room': staff.office_room,
                'current_status': staff.current_status,
                'status_message': staff.status_message,
                'whatsapp_enabled': staff.whatsapp_enabled,
                'is_active': staff.is_active,
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Data JSON tidak valid'
        })
    except Exception as e:
        logger.error(f"Error creating staff: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Terjadi kesalahan sistem: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["PUT"])
def update_staff(request, staff_id):
    """Update existing staff member"""
    try:
        staff = get_object_or_404(Staff, id=staff_id)
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['name', 'position', 'department', 'phone_number']
        for field in required_fields:
            if not data.get(field):
                return JsonResponse({
                    'status': 'error',
                    'message': f'Field {field} wajib diisi'
                })
        
        # Validate phone number format
        phone_number = data['phone_number'].strip()
        if not re.match(r'^(\+62|62|0)[0-9]{8,12}$', phone_number):
            return JsonResponse({
                'status': 'error',
                'message': 'Format nomor telepon tidak valid. Gunakan format: 08xxxx, 62xxxx, atau +62xxxx'
            })
        
        # Validate email if provided
        email = data.get('email', '').strip()
        if email:
            try:
                validate_email(email)
            except ValidationError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Format email tidak valid'
                })
        
        # Validate department
        valid_departments = [choice[0] for choice in Staff.DEPARTMENTS]
        if data['department'] not in valid_departments:
            return JsonResponse({
                'status': 'error',
                'message': 'Departemen tidak valid'
            })
        
        # Validate status
        valid_statuses = [choice[0] for choice in Staff.STATUS_CHOICES]
        current_status = data.get('current_status', 'available')
        if current_status not in valid_statuses:
            return JsonResponse({
                'status': 'error',
                'message': 'Status tidak valid'
            })
        
        # Check if phone number already exists for other staff
        existing_phone = Staff.objects.filter(phone_number=phone_number).exclude(id=staff_id)
        if existing_phone.exists():
            return JsonResponse({
                'status': 'error',
                'message': 'Nomor telepon sudah terdaftar untuk staff lain'
            })
        
        # Check if email already exists for other staff (if provided)
        if email:
            existing_email = Staff.objects.filter(email=email).exclude(id=staff_id)
            if existing_email.exists():
                return JsonResponse({
                    'status': 'error',
                    'message': 'Email sudah terdaftar untuk staff lain'
                })
        
        # Store old values for comparison
        old_name = staff.name
        old_status = staff.current_status
        old_whatsapp = staff.whatsapp_enabled
        
        # Update staff
        staff.name = data['name'].strip()
        staff.position = data['position'].strip()
        staff.department = data['department']
        staff.phone_number = phone_number
        staff.email = email if email else None
        staff.office_room = data.get('office_room', '').strip() or None
        staff.current_status = current_status
        staff.status_message = data.get('status_message', '').strip() or None
        staff.whatsapp_enabled = data.get('whatsapp_enabled', True)
        staff.is_active = data.get('is_active', True)
        
        staff.save()
        
        # Create notification for significant changes
        changes = []
        if old_name != staff.name:
            changes.append(f'nama dari {old_name} ke {staff.name}')
        if old_status != staff.current_status:
            old_status_display = dict(Staff.STATUS_CHOICES).get(old_status, old_status)
            new_status_display = dict(Staff.STATUS_CHOICES).get(staff.current_status, staff.current_status)
            changes.append(f'status dari {old_status_display} ke {new_status_display}')
        if old_whatsapp != staff.whatsapp_enabled:
            whatsapp_status = 'diaktifkan' if staff.whatsapp_enabled else 'dinonaktifkan'
            changes.append(f'WhatsApp {whatsapp_status}')
        
        if changes:
            try:
                create_notification(
                    guest=None,  # System notification
                    notification_type='staff_updated',
                    title='Data Staff Diperbarui',
                    message=f'Data staff {staff.name} telah diperbarui: {", ".join(changes)}'
                )
            except:
                pass
        
        logger.info(f"Staff updated: {staff.name} (ID: {staff.id})")
        
        return JsonResponse({
            'status': 'success',
            'message': 'Data staff berhasil diperbarui',
            'data': {
                'id': staff.id,
                'name': staff.name,
                'position': staff.position,
                'department': staff.department,
                'department_display': staff.department_display,
                'phone_number': staff.phone_number,
                'email': staff.email,
                'office_room': staff.office_room,
                'current_status': staff.current_status,
                'status_message': staff.status_message,
                'whatsapp_enabled': staff.whatsapp_enabled,
                'is_active': staff.is_active,
            }
        })
        
    except Staff.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Staff tidak ditemukan'
        })
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Data JSON tidak valid'
        })
    except Exception as e:
        logger.error(f"Error updating staff {staff_id}: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Terjadi kesalahan sistem: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_staff(request, staff_id):
    """Delete staff member"""
    try:
        staff = get_object_or_404(Staff, id=staff_id)
        
        # Check if staff has active visits
        active_visits = VisitLog.objects.filter(
            staff_to_meet=staff,
            check_out_time__isnull=True
        ).count()
        
        if active_visits > 0:
            return JsonResponse({
                'status': 'error',
                'message': f'Tidak dapat menghapus staff karena masih memiliki {active_visits} kunjungan aktif'
            })
        
        # Store staff info for notification
        staff_name = staff.name
        staff_position = staff.position
        
        # Soft delete: mark as inactive instead of actual deletion
        # This preserves historical data
        staff.is_active = False
        staff.whatsapp_enabled = False
        staff.current_status = 'off'
        staff.status_message = 'Staff telah dihapus dari sistem'
        staff.save()
        
        # Create notification
        try:
            create_notification(
                guest=None,  # System notification
                notification_type='staff_deleted',
                title='Staff Dihapus',
                message=f'Staff {staff_name} ({staff_position}) telah dihapus dari sistem'
            )
        except:
            pass
        
        logger.info(f"Staff soft deleted: {staff_name} (ID: {staff_id})")
        
        return JsonResponse({
            'status': 'success',
            'message': f'Staff {staff_name} berhasil dihapus'
        })
        
    except Staff.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Staff tidak ditemukan'
        })
    except Exception as e:
        logger.error(f"Error deleting staff {staff_id}: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Terjadi kesalahan sistem: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["GET"])
def get_staff_detail(request, staff_id):
    """Get detailed staff information"""
    try:
        staff = get_object_or_404(Staff, id=staff_id)
        
        # Get related statistics
        total_visits = VisitLog.objects.filter(staff_to_meet=staff).count()
        active_visits = VisitLog.objects.filter(
            staff_to_meet=staff,
            check_out_time__isnull=True
        ).count()
        
        # Get recent visits
        recent_visits = VisitLog.objects.filter(
            staff_to_meet=staff
        ).order_by('-check_in_time')[:5]
        
        recent_visits_data = []
        for visit in recent_visits:
            recent_visits_data.append({
                'id': visit.id,
                'guest_name': visit.guest.name,
                'visit_purpose': visit.get_visit_purpose_display(),
                'check_in_time': visit.check_in_time.strftime('%d/%m/%Y %H:%M'),
                'check_out_time': visit.check_out_time.strftime('%d/%m/%Y %H:%M') if visit.check_out_time else None,
                'duration_minutes': visit.duration_minutes,
                'urgency_level': visit.get_urgency_level_display()
            })
        
        return JsonResponse({
            'status': 'success',
            'data': {
                'id': staff.id,
                'name': staff.name,
                'position': staff.position,
                'department': staff.department,
                'department_display': staff.department_display,
                'phone_number': staff.phone_number,
                'email': staff.email,
                'office_room': staff.office_room,
                'current_status': staff.current_status,
                'status_display': staff.get_current_status_display(),
                'status_message': staff.status_message,
                'whatsapp_enabled': staff.whatsapp_enabled,
                'is_active': staff.is_active,
                'created_at': staff.created_at.isoformat(),
                'updated_at': staff.updated_at.isoformat(),
                'statistics': {
                    'total_visits': total_visits,
                    'active_visits': active_visits,
                },
                'recent_visits': recent_visits_data
            }
        })
        
    except Staff.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Staff tidak ditemukan'
        })
    except Exception as e:
        logger.error(f"Error getting staff detail {staff_id}: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Terjadi kesalahan sistem: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def bulk_update_staff_status(request):
    """Bulk update staff status"""
    try:
        data = json.loads(request.body)
        staff_ids = data.get('staff_ids', [])
        new_status = data.get('status')
        status_message = data.get('status_message', '')
        
        if not staff_ids:
            return JsonResponse({
                'status': 'error',
                'message': 'Tidak ada staff yang dipilih'
            })
        
        if not new_status:
            return JsonResponse({
                'status': 'error',
                'message': 'Status baru wajib diisi'
            })
        
        # Validate status
        valid_statuses = [choice[0] for choice in Staff.STATUS_CHOICES]
        if new_status not in valid_statuses:
            return JsonResponse({
                'status': 'error',
                'message': 'Status tidak valid'
            })
        
        # Update staff
        updated_count = Staff.objects.filter(
            id__in=staff_ids,
            is_active=True
        ).update(
            current_status=new_status,
            status_message=status_message.strip() if status_message else None,
            updated_at=timezone.now()
        )
        
        if updated_count > 0:
            # Create notification
            status_display = dict(Staff.STATUS_CHOICES).get(new_status, new_status)
            try:
                create_notification(
                    guest=None,
                    notification_type='staff_bulk_update',
                    title='Update Status Staff Massal',
                    message=f'{updated_count} staff telah diubah statusnya menjadi {status_display}'
                )
            except:
                pass
            
            logger.info(f"Bulk status update: {updated_count} staff updated to {new_status}")
        
        return JsonResponse({
            'status': 'success',
            'message': f'{updated_count} staff berhasil diperbarui',
            'updated_count': updated_count
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Data JSON tidak valid'
        })
    except Exception as e:
        logger.error(f"Error bulk updating staff status: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Terjadi kesalahan sistem: {str(e)}'
        })

# Update the existing get_staff_list function to include more details
def get_staff_list(request):
    """API endpoint to get list of staff with enhanced information"""
    try:
        # Get query parameters
        include_inactive = request.GET.get('include_inactive', 'false').lower() == 'true'
        department = request.GET.get('department', '')
        status = request.GET.get('status', '')
        whatsapp_only = request.GET.get('whatsapp_only', 'false').lower() == 'true'
        
        # Build query
        query = Staff.objects.all()
        
        if not include_inactive:
            query = query.filter(is_active=True)
        
        if department:
            query = query.filter(department=department)
        
        if status:
            query = query.filter(current_status=status)
        
        if whatsapp_only:
            query = query.filter(whatsapp_enabled=True)
        
        query = query.order_by('department', 'name')
        
        staff_list = []
        for staff in query:
            # Get visit statistics
            total_visits = VisitLog.objects.filter(staff_to_meet=staff).count()
            active_visits = VisitLog.objects.filter(
                staff_to_meet=staff,
                check_out_time__isnull=True
            ).count()
            
            staff_data = {
                'id': staff.id,
                'name': staff.name,
                'position': staff.position,
                'department': staff.department,
                'department_display': staff.department_display,
                'current_status': staff.current_status,
                'status_display': staff.get_current_status_display(),
                'status_message': staff.status_message,
                'office_room': staff.office_room,
                'phone_number': staff.phone_number,
                'email': staff.email,
                'photo': staff.photo.url if staff.photo else None,
                'whatsapp_enabled': staff.whatsapp_enabled,
                'is_active': staff.is_active,
                'created_at': staff.created_at.isoformat(),
                'updated_at': staff.updated_at.isoformat(),
                'statistics': {
                    'total_visits': total_visits,
                    'active_visits': active_visits,
                }
            }
            staff_list.append(staff_data)
        
        return JsonResponse({
            'status': 'success',
            'data': staff_list,
            'total_count': len(staff_list),
            'filters_applied': {
                'include_inactive': include_inactive,
                'department': department,
                'status': status,
                'whatsapp_only': whatsapp_only
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting staff list: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
        
@csrf_exempt
@require_http_methods(["POST"])
def create_version(request):
    """Create a new version with changelog items"""
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        if not data.get('version_number') or not data.get('title'):
            return JsonResponse({
                'status': 'error',
                'message': 'Version number and title are required'
            }, status=400)
        
        # Check if version number already exists
        if Version.objects.filter(version_number=data['version_number']).exists():
            return JsonResponse({
                'status': 'error',
                'message': f'Version {data["version_number"]} already exists'
            }, status=400)
        
        # Create version
        version = Version.objects.create(
            version_number=data['version_number'],
            title=data['title'],
            description=data.get('description', ''),
            is_current=data.get('is_current', False),
            release_date=timezone.now()
        )
        
        # Create changelog items
        changelog_items = data.get('changelog_items', [])
        for item_data in changelog_items:
            ChangelogItem.objects.create(
                version=version,
                item_type=item_data['item_type'],
                title=item_data['title'],
                description=item_data['description'],
                is_highlighted=item_data.get('is_highlighted', False)
            )
        
        return JsonResponse({
            'status': 'success',
            'message': f'Version {version.version_number} created successfully',
            'data': {
                'id': version.id,
                'version_number': version.version_number,
                'title': version.title,
                'is_current': version.is_current
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON data'
        }, status=400)
    
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["PUT"])
def update_version(request, version_id):
    """Update an existing version"""
    try:
        version = get_object_or_404(Version, id=version_id)
        data = json.loads(request.body)
        
        # Update version fields
        version.version_number = data.get('version_number', version.version_number)
        version.title = data.get('title', version.title)
        version.description = data.get('description', version.description)
        
        # Handle current version logic
        if data.get('is_current', False) and not version.is_current:
            version.is_current = True
        
        version.save()
        
        # Update changelog items - delete existing and create new ones
        version.changelog_items.all().delete()
        
        changelog_items = data.get('changelog_items', [])
        for item_data in changelog_items:
            ChangelogItem.objects.create(
                version=version,
                item_type=item_data['item_type'],
                title=item_data['title'],
                description=item_data['description'],
                is_highlighted=item_data.get('is_highlighted', False)
            )
        
        return JsonResponse({
            'status': 'success',
            'message': f'Version {version.version_number} updated successfully'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON data'
        }, status=400)
    
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["DELETE"])
def delete_version(request, version_id):
    """Delete a version (only if not current)"""
    try:
        version = get_object_or_404(Version, id=version_id)
        
        # Prevent deletion of current version
        if version.is_current:
            return JsonResponse({
                'status': 'error',
                'message': 'Cannot delete the current version'
            }, status=400)
        
        version_number = version.version_number
        version.delete()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Version {version_number} deleted successfully'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


def get_versions_list(request):
    """Get list of all versions with changelog items"""
    try:
        versions = Version.objects.all().order_by('-release_date')
        current_version = versions.filter(is_current=True).first()
        
        versions_data = []
        for version in versions:
            changelog_items = []
            for item in version.changelog_items.all():
                changelog_items.append({
                    'id': item.id,
                    'item_type': item.item_type,
                    'title': item.title,
                    'description': item.description,
                    'is_highlighted': item.is_highlighted,
                    'created_at': item.created_at.isoformat()
                })
            
            versions_data.append({
                'id': version.id,
                'version_number': version.version_number,
                'title': version.title,
                'description': version.description,
                'release_date': version.release_date.isoformat(),
                'is_current': version.is_current,
                'created_at': version.created_at.isoformat(),
                'changelog_items': changelog_items
            })
        
        current_version_data = None
        if current_version:
            current_version_data = {
                'id': current_version.id,
                'version_number': current_version.version_number,
                'title': current_version.title,
                'description': current_version.description,
                'release_date': current_version.release_date.isoformat()
            }
        
        return JsonResponse({
            'status': 'success',
            'data': {
                'current_version': current_version_data,
                'versions': versions_data
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


# Juga update view changelog_data yang sudah ada untuk memastikan konsistensi
def changelog_data(request):
    """Enhanced changelog data view"""
    try:
        # Get current version
        current_version = Version.objects.filter(is_current=True).first()
        
        # Get all versions ordered by release date (newest first)
        versions = Version.objects.all().order_by('-release_date')
        
        versions_data = []
        for version in versions:
            # Get changelog items for this version
            changelog_items = []
            for item in version.changelog_items.all().order_by('item_type', 'created_at'):
                changelog_items.append({
                    'id': item.id,
                    'item_type': item.item_type,
                    'title': item.title,
                    'description': item.description,
                    'is_highlighted': item.is_highlighted,
                })
            
            versions_data.append({
                'id': version.id,
                'version_number': version.version_number,
                'title': version.title,
                'description': version.description,
                'release_date': version.release_date.isoformat(),
                'is_current': version.is_current,
                'changelog_items': changelog_items
            })
        
        # Prepare current version data
        current_version_data = None
        if current_version:
            current_version_data = {
                'id': current_version.id,
                'version_number': current_version.version_number,
                'title': current_version.title,
                'description': current_version.description,
                'release_date': current_version.release_date.isoformat()
            }
        
        return JsonResponse({
            'status': 'success',
            'data': {
                'current_version': current_version_data,
                'versions': versions_data
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)