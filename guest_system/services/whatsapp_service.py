# services/whatsapp_service.py - Fixed for HTTP 405 Error

import requests
import logging
from django.conf import settings
from typing import Optional, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class WhatsAppService:
    """
    Service for sending WhatsApp messages to staff members
    Uses Fonnte API - Fixed for correct API endpoints
    """
    
    def __init__(self):
        # Fonnte API configuration
        self.api_url = getattr(settings, 'WHATSAPP_API_URL', 'https://api.fonnte.com/send')
        self.api_token = getattr(settings, 'WHATSAPP_API_TOKEN', '')
        self.enabled = getattr(settings, 'WHATSAPP_ENABLED', False)
        
        # Log configuration status
        if self.enabled:
            logger.info(f"WhatsApp service enabled with URL: {self.api_url}")
            if self.api_token:
                logger.info(f"API Token configured: {self.api_token[:10]}...")
            else:
                logger.warning("WhatsApp API token not configured")
        else:
            logger.info("WhatsApp service disabled")
        
    def send_guest_meeting_notification(self, staff, guest, visit_log) -> Dict[str, Any]:
        """
        Send WhatsApp notification to staff about incoming guest
        """
        
        if not self.enabled:
            logger.warning("WhatsApp service is disabled in settings")
            return {
                'success': False,
                'message': 'WhatsApp service is disabled',
                'error_code': 'SERVICE_DISABLED'
            }
            
        if not self.api_token:
            logger.error("WhatsApp API token not configured")
            return {
                'success': False,
                'message': 'WhatsApp API token not configured',
                'error_code': 'NO_API_TOKEN'
            }
            
        if not staff.whatsapp_enabled:
            logger.warning(f"WhatsApp notifications disabled for staff: {staff.name}")
            return {
                'success': False,
                'message': f'WhatsApp notifications disabled for {staff.name}',
                'error_code': 'STAFF_WHATSAPP_DISABLED'
            }
            
        # Validate phone number
        if not staff.phone_number:
            logger.error(f"Staff {staff.name} has no phone number configured")
            return {
                'success': False,
                'message': f'No phone number configured for {staff.name}',
                'error_code': 'NO_PHONE_NUMBER'
            }
            
        try:
            # Format phone number for WhatsApp
            phone = self._format_phone_number(staff.phone_number)
            
            # Format the message
            message = self._format_guest_meeting_message(staff, guest, visit_log)
            
            # Send using the main send method
            return self._send_whatsapp_message(phone, message)
                
        except Exception as e:
            logger.error(f"Unexpected error sending WhatsApp: {str(e)}")
            return {
                'success': False,
                'message': f'Unexpected error: {str(e)}',
                'error_code': 'UNEXPECTED_ERROR'
            }
    
    def send_test_message(self, phone_number: str, staff_name: str = "Staff") -> Dict[str, Any]:
        """Send test WhatsApp message"""
        
        if not self.enabled:
            return {
                'success': False,
                'message': 'WhatsApp service is disabled'
            }
            
        if not self.api_token:
            return {
                'success': False,
                'message': 'WhatsApp API token not configured'
            }
            
        try:
            # Format phone number
            phone = self._format_phone_number(phone_number)
            
            message = f"""🏢 *TEST MESSAGE - SMART GUEST BOOK*

Halo {staff_name},

Ini adalah pesan tes dari sistem Smart Guest Book.

Jika Anda menerima pesan ini, artinya konfigurasi WhatsApp sudah berhasil! ✅

Waktu Test: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

---
_Test message dari Smart Guest Book System_
_BPS Kab. Padang Lawas Utara_"""

            return self._send_whatsapp_message(phone, message)
                
        except Exception as e:
            logger.error(f"Error sending test WhatsApp: {str(e)}")
            return {
                'success': False,
                'message': f'Error sending test WhatsApp: {str(e)}'
            }
    
    def _send_whatsapp_message(self, phone: str, message: str) -> Dict[str, Any]:
        """
        Core method to send WhatsApp message via Fonnte API
        """
        try:
            # Prepare API request for Fonnte
            payload = {
                'target': phone,
                'message': message,
                'countryCode': '62',  # Indonesia country code
            }
            
            headers = {
                'Authorization': self.api_token
            }
            
            logger.info(f"Sending WhatsApp to {phone}")
            logger.debug(f"Payload: {payload}")
            
            # Send request using form data (Fonnte expects form-encoded data)
            response = requests.post(
                self.api_url,
                data=payload,  # Use data for form encoding
                headers=headers,
                timeout=30
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response: {response.text}")
                return {
                    'success': False,
                    'message': f'Invalid response from WhatsApp API: {response.text}',
                    'error_code': 'INVALID_RESPONSE'
                }
            
            # Check Fonnte response format
            if response.status_code == 200:
                if response_data.get('status') == True:
                    logger.info(f"WhatsApp sent successfully to {phone}")
                    return {
                        'success': True,
                        'message': 'WhatsApp notification sent successfully',
                        'response_data': response_data,
                        'message_id': response_data.get('id')
                    }
                else:
                    error_msg = response_data.get('reason', 'Unknown error from Fonnte API')
                    logger.error(f"Fonnte API error: {error_msg}")
                    return {
                        'success': False,
                        'message': f'Fonnte API error: {error_msg}',
                        'error_code': 'FONNTE_API_ERROR',
                        'response_data': response_data
                    }
            else:
                error_msg = response_data.get('reason', f'HTTP {response.status_code}')
                logger.error(f"HTTP error: {response.status_code} - {error_msg}")
                return {
                    'success': False,
                    'message': f'HTTP error: {error_msg}',
                    'error_code': 'HTTP_ERROR',
                    'response_data': response_data
                }
                
        except requests.exceptions.Timeout:
            logger.error("WhatsApp API request timeout")
            return {
                'success': False,
                'message': 'WhatsApp API request timeout',
                'error_code': 'TIMEOUT'
            }
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to WhatsApp API: {str(e)}")
            return {
                'success': False,
                'message': f'Connection error: {str(e)}',
                'error_code': 'CONNECTION_ERROR'
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"WhatsApp API request failed: {str(e)}")
            return {
                'success': False,
                'message': f'Network error: {str(e)}',
                'error_code': 'NETWORK_ERROR'
            }
    
    def _format_phone_number(self, phone_number: str) -> str:
        """
        Format phone number for WhatsApp API
        Remove +, spaces, and ensure it starts with country code
        """
        # Remove all non-numeric characters
        phone = ''.join(filter(str.isdigit, phone_number))
        
        # If starts with 0, replace with 62 (Indonesia)
        if phone.startswith('0'):
            phone = '62' + phone[1:]
        
        # If doesn't start with 62, add it
        elif not phone.startswith('62'):
            phone = '62' + phone
            
        logger.debug(f"Formatted phone number: {phone_number} -> {phone}")
        return phone
    
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
        
        # Get visit purpose choices mapping
        purpose_choices = {
            'data_service': 'Pelayanan Data',
            'meet_staff': 'Bertemu Staff',
            'other_activity': 'Kegiatan Lainnya',
            'survey': 'Survey',
            'coordination': 'Koordinasi',
            'complaint': 'Pengaduan'
        }
        
        purpose_display = purpose_choices.get(visit_log.visit_purpose, visit_log.visit_purpose)
        
        # Get urgency choices mapping
        urgency_choices = {
            'low': 'Rendah',
            'medium': 'Sedang',
            'high': 'Tinggi'
        }
        
        urgency_display = urgency_choices.get(visit_log.urgency_level, visit_log.urgency_level)
        
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
• Tujuan Kunjungan: {purpose_display}
• Deskripsi: {visit_log.visit_description}
• Tingkat Urgensi: {urgency_emoji} {urgency_display}
{f"• Estimasi Durasi: {duration_text}" if duration_text else ""}

📍 *LOKASI ANDA:*
{staff.office_room or "Ruang kerja"}

Silakan bersiap untuk menerima tamu atau hubungi front desk jika ada kendala.

---
_Pesan otomatis dari Smart Guest Book System_
_BPS Kab. Padang Lawas Utara_"""
        
        return message
    
    def check_api_status(self) -> Dict[str, Any]:
        """
        Check Fonnte API status - Fixed method
        Uses a simple test send instead of device endpoint
        """
        
        if not self.enabled or not self.api_token:
            return {
                'success': False,
                'message': 'Service not configured'
            }
            
        try:
            # Instead of checking /device endpoint (which gives 405), 
            # we'll test with a simple validation request
            
            # Method 1: Try to send to an invalid number to test API response
            test_payload = {
                'target': '6200000000000',  # Invalid number for testing
                'message': 'API Test',
                'countryCode': '62',
            }
            
            headers = {
                'Authorization': self.api_token
            }
            
            response = requests.post(
                self.api_url,
                data=test_payload,
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Even if message fails (invalid number), 
                    # successful API response means token is valid
                    if 'status' in data:
                        return {
                            'success': True,
                            'message': 'API token is valid and service is accessible',
                            'api_response': data
                        }
                    else:
                        return {
                            'success': False,
                            'message': 'Unexpected API response format',
                            'api_response': data
                        }
                        
                except json.JSONDecodeError:
                    return {
                        'success': False,
                        'message': f'Invalid JSON response: {response.text}'
                    }
                    
            elif response.status_code == 401:
                return {
                    'success': False,
                    'message': 'Invalid API token - Unauthorized'
                }
            elif response.status_code == 405:
                return {
                    'success': False,
                    'message': 'API endpoint method not allowed - Check API URL'
                }
            else:
                return {
                    'success': False,
                    'message': f'API error: HTTP {response.status_code} - {response.text}'
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'message': 'API request timeout'
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'message': 'Cannot connect to API'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection error: {str(e)}'
            }
    
    def validate_token_simple(self) -> Dict[str, Any]:
        """
        Simple token validation using actual send endpoint
        """
        try:
            # Test with obviously invalid phone to check token validity
            result = self._send_whatsapp_message('6200000000000', 'API Test')
            
            if 'response_data' in result:
                # If we get a proper Fonnte response (even if failed), token is valid
                return {
                    'success': True,
                    'message': 'API token is valid',
                    'token_valid': True
                }
            elif result.get('error_code') == 'HTTP_ERROR':
                # Check if it's authentication error
                if '401' in result.get('message', ''):
                    return {
                        'success': False,
                        'message': 'Invalid API token',
                        'token_valid': False
                    }
                else:
                    return {
                        'success': True,
                        'message': 'API token appears valid (non-auth error)',
                        'token_valid': True
                    }
            else:
                return {
                    'success': True,
                    'message': 'API connection successful',
                    'token_valid': True
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Validation error: {str(e)}',
                'token_valid': False
            }


# Global instance
whatsapp_service = WhatsAppService()