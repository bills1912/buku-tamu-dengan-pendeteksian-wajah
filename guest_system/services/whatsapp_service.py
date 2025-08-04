# services/whatsapp_service.py

import requests
import logging
from django.conf import settings
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class WhatsAppService:
    """
    Service for sending WhatsApp messages to staff members
    Uses Fonnte API or similar WhatsApp Gateway service
    """
    
    def __init__(self):
        # You can configure these in settings.py
        self.api_url = getattr(settings, 'WHATSAPP_API_URL', 'https://api.fonnte.com/send')
        self.api_token = getattr(settings, 'WHATSAPP_API_TOKEN', '')
        self.enabled = getattr(settings, 'WHATSAPP_ENABLED', False)
        
    def send_guest_meeting_notification(self, staff, guest, visit_log) -> Dict[str, Any]:
        """
        Send WhatsApp notification to staff about incoming guest
        
        Args:
            staff: Staff object
            guest: Guest object  
            visit_log: VisitLog object
            
        Returns:
            dict: Response with status and message
        """
        
        if not self.enabled:
            return {
                'success': False,
                'message': 'WhatsApp service is disabled',
                'error_code': 'SERVICE_DISABLED'
            }
            
        if not self.api_token:
            return {
                'success': False,
                'message': 'WhatsApp API token not configured',
                'error_code': 'NO_API_TOKEN'
            }
            
        if not staff.whatsapp_enabled:
            return {
                'success': False,
                'message': f'WhatsApp notifications disabled for {staff.name}',
                'error_code': 'STAFF_WHATSAPP_DISABLED'
            }
            
        try:
            # Format the message
            message = self._format_guest_meeting_message(staff, guest, visit_log)
            
            # Prepare API request
            payload = {
                'target': staff.whatsapp_number,
                'message': message,
                'countryCode': '62',  # Indonesia country code
            }
            
            headers = {
                'Authorization': self.api_token,
                'Content-Type': 'application/json'
            }
            
            # Send request
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('status'):
                logger.info(f"WhatsApp sent successfully to {staff.name} ({staff.whatsapp_number})")
                return {
                    'success': True,
                    'message': 'WhatsApp notification sent successfully',
                    'response_data': response_data
                }
            else:
                error_msg = response_data.get('reason', 'Unknown error')
                logger.error(f"WhatsApp send failed: {error_msg}")
                return {
                    'success': False,
                    'message': f'Failed to send WhatsApp: {error_msg}',
                    'error_code': 'API_ERROR',
                    'response_data': response_data
                }
                
        except requests.exceptions.Timeout:
            logger.error("WhatsApp API request timeout")
            return {
                'success': False,
                'message': 'WhatsApp API request timeout',
                'error_code': 'TIMEOUT'
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"WhatsApp API request failed: {str(e)}")
            return {
                'success': False,
                'message': f'Network error: {str(e)}',
                'error_code': 'NETWORK_ERROR'
            }
            
        except Exception as e:
            logger.error(f"Unexpected error sending WhatsApp: {str(e)}")
            return {
                'success': False,
                'message': f'Unexpected error: {str(e)}',
                'error_code': 'UNEXPECTED_ERROR'
            }
    
    def _format_guest_meeting_message(self, staff, guest, visit_log) -> str:
        """Format the WhatsApp message for staff notification"""
        
        # Get urgency emoji
        urgency_emoji = {
            'low': '🟢',
            'medium': '🟡', 
            'high': '🔴'
        }.get(visit_log.urgency_level, '🟢')
        
        # Format expected duration
        duration_text = ""
        if visit_log.expected_duration:
            if visit_log.expected_duration < 60:
                duration_text = f"~{visit_log.expected_duration} menit"
            else:
                hours = visit_log.expected_duration // 60
                minutes = visit_log.expected_duration % 60
                if minutes > 0:
                    duration_text = f"~{hours} jam {minutes} menit"
                else:
                    duration_text = f"~{hours} jam"
        
        # Format check-in time
        check_in_time = visit_log.check_in_time.strftime('%H:%M')
        check_in_date = visit_log.check_in_time.strftime('%d/%m/%Y')
        
        message = f"""🏢 *SMART GUEST BOOK - NOTIFIKASI TAMU*

👋 Halo {staff.name},

Ada tamu yang ingin bertemu dengan Anda:

📋 *INFORMASI TAMU:*
• Nama: {guest.name}
• Perusahaan: {guest.company or "Tidak disebutkan"}
• Telepon: {guest.phone or "Tidak disebutkan"}
• Email: {guest.email or "Tidak disebutkan"}

⏰ *DETAIL KUNJUNGAN:*
• Waktu Check-in: {check_in_time}, {check_in_date}
• Tujuan: {visit_log.visit_description}
• Tingkat Urgensi: {urgency_emoji} {dict(visit_log.URGENCY_CHOICES)[visit_log.urgency_level]}
{f"• Estimasi Durasi: {duration_text}" if duration_text else ""}

📍 *LOKASI ANDA:*
{staff.office_room or "Ruang kerja"}

Silakan bersiap untuk menerima tamu atau hubungi front desk jika ada kendala.

---
_Pesan otomatis dari Smart Guest Book System_
_BPS Kab. Sijunjung_"""
        
        return message
    
    def send_test_message(self, phone_number: str, staff_name: str = "Staff") -> Dict[str, Any]:
        """Send test WhatsApp message"""
        
        if not self.enabled:
            return {
                'success': False,
                'message': 'WhatsApp service is disabled'
            }
            
        try:
            message = f"""🏢 *TEST MESSAGE - SMART GUEST BOOK*

Halo {staff_name},

Ini adalah pesan tes dari sistem Smart Guest Book.

Jika Anda menerima pesan ini, artinya konfigurasi WhatsApp sudah berhasil! ✅

---
_Test message dari Smart Guest Book System_
_BPS Kab. Sijunjung_"""

            payload = {
                'target': phone_number,
                'message': message,
                'countryCode': '62',
            }
            
            headers = {
                'Authorization': self.api_token,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('status'):
                return {
                    'success': True,
                    'message': 'Test WhatsApp sent successfully',
                    'response_data': response_data
                }
            else:
                return {
                    'success': False,
                    'message': f'Failed to send test WhatsApp: {response_data.get("reason", "Unknown error")}',
                    'response_data': response_data
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error sending test WhatsApp: {str(e)}'
            }


# Global instance
whatsapp_service = WhatsAppService()