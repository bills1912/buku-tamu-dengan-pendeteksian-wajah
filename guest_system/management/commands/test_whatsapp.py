# guest_system/management/commands/test_whatsapp.py

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from guest_system.models import Staff, Guest, VisitLog
from guest_system.services.whatsapp_service import whatsapp_service
from django.utils import timezone
import requests
import json
import sys
import traceback

class Command(BaseCommand):
    help = 'Test WhatsApp integration and send test messages'

    def add_arguments(self, parser):
        parser.add_argument(
            '--staff-id',
            type=int,
            help='Staff ID to send test message to'
        )
        parser.add_argument(
            '--phone',
            type=str,
            help='Phone number to send test message to (format: 6281234567890)'
        )
        parser.add_argument(
            '--check-api',
            action='store_true',
            help='Check API status and device connection'
        )
        parser.add_argument(
            '--list-staff',
            action='store_true',
            help='List all staff with WhatsApp enabled'
        )
        parser.add_argument(
            '--test-guest-notification',
            action='store_true',
            help='Test guest meeting notification with sample data'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Run all tests (comprehensive test suite)'
        )
        parser.add_argument(
            '--format-phone',
            type=str,
            help='Test phone number formatting'
        )
        parser.add_argument(
            '--interactive',
            action='store_true',
            help='Interactive mode for testing'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output with detailed information'
        )
        parser.add_argument(
            '--test-all-staff',
            action='store_true',
            help='Send test message to all WhatsApp-enabled staff'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Dry run mode - show what would be sent without actually sending'
        )

    def handle(self, *args, **options):
        self.verbose = options.get('verbose', False)
        self.dry_run = options.get('dry_run', False)
        
        if self.dry_run:
            self.stdout.write(self.style.WARNING('🏃 DRY RUN MODE - No messages will be sent'))
        
        self.stdout.write(self.style.SUCCESS('🚀 Starting WhatsApp Test Suite'))
        self.stdout.write('=' * 60)
        
        # Check configuration first
        config_ok = self.check_configuration()
        
        # Handle different options
        try:
            if options['all']:
                self.run_all_tests()
            elif options['interactive']:
                self.interactive_mode()
            elif options['check_api']:
                self.check_api_status()
            elif options['list_staff']:
                self.list_staff()
            elif options['staff_id']:
                self.test_staff_message(options['staff_id'])
            elif options['phone']:
                self.test_phone_message(options['phone'])
            elif options['test_guest_notification']:
                self.test_guest_notification()
            elif options['format_phone']:
                self.test_phone_formatting(options['format_phone'])
            elif options['test_all_staff']:
                self.test_all_staff()
            else:
                self.show_help()
        except KeyboardInterrupt:
            self.stdout.write('\n\n⏹️  Operation cancelled by user')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\n❌ Unexpected error: {str(e)}'))
            if self.verbose:
                self.stdout.write(traceback.format_exc())
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        self.stdout.write(self.style.SUCCESS('\n🔄 Running All Tests...'))
        
        tests = [
            ('Configuration Check', self.check_configuration),
            ('API Status Check', self.check_api_status),
            ('Staff List', self.list_staff),
            ('Phone Formatting Test', lambda: self.test_phone_formatting('081234567890')),
        ]
        
        # Only add guest notification test if not dry run
        if not self.dry_run:
            tests.append(('Guest Notification Test', self.test_guest_notification))
        
        results = []
        
        for test_name, test_func in tests:
            self.stdout.write(f'\n--- {test_name} ---')
            try:
                success = test_func()
                if success is not False:  # None or True considered success
                    results.append((test_name, '✅', 'Passed'))
                else:
                    results.append((test_name, '❌', 'Failed'))
            except Exception as e:
                results.append((test_name, '❌', str(e)))
                self.stdout.write(self.style.ERROR(f'   Error: {str(e)}'))
        
        # Summary
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write(self.style.SUCCESS('📊 TEST SUMMARY'))
        self.stdout.write('=' * 60)
        
        for test_name, status, message in results:
            self.stdout.write(f'{status} {test_name}: {message}')
        
        passed = sum(1 for _, status, _ in results if status == '✅')
        total = len(results)
        
        if passed == total:
            self.stdout.write(self.style.SUCCESS(f'\n🎉 All tests passed! ({passed}/{total})'))
        else:
            self.stdout.write(self.style.WARNING(f'\n⚠️  {passed}/{total} tests passed'))
        
        return passed == total
    
    def interactive_mode(self):
        """Interactive testing mode"""
        self.stdout.write(self.style.SUCCESS('\n🎮 Interactive WhatsApp Testing Mode'))
        self.stdout.write('Type "help" for available commands, "quit" to exit\n')
        
        while True:
            try:
                command = input('WhatsApp Test > ').strip()
                
                if not command:
                    continue
                    
                cmd_parts = command.lower().split()
                cmd = cmd_parts[0]
                
                if cmd in ['quit', 'exit', 'q']:
                    break
                elif cmd == 'help':
                    self.show_interactive_help()
                elif cmd == 'config':
                    self.check_configuration()
                elif cmd == 'api':
                    self.check_api_status()
                elif cmd == 'staff':
                    self.list_staff()
                elif cmd == 'test' and len(cmd_parts) > 1:
                    try:
                        staff_id = int(cmd_parts[1])
                        self.test_staff_message(staff_id)
                    except ValueError:
                        self.stdout.write(self.style.ERROR('Invalid staff ID. Use: test <number>'))
                elif cmd == 'phone' and len(cmd_parts) > 1:
                    phone = cmd_parts[1]
                    self.test_phone_message(phone)
                elif cmd == 'guest':
                    self.test_guest_notification()
                elif cmd == 'format' and len(cmd_parts) > 1:
                    phone = cmd_parts[1]
                    self.test_phone_formatting(phone)
                elif cmd == 'all':
                    self.run_all_tests()
                elif cmd == 'testall':
                    self.test_all_staff()
                elif cmd == 'clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                else:
                    self.stdout.write(self.style.WARNING('Unknown command. Type "help" for available commands.'))
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error: {str(e)}'))
        
        self.stdout.write('\nGoodbye! 👋')
    
    def show_interactive_help(self):
        """Show interactive mode help"""
        commands = [
            ('help', 'Show this help message'),
            ('config', 'Check WhatsApp configuration'),
            ('api', 'Check API status'),
            ('staff', 'List all staff'),
            ('test <id>', 'Test message to staff ID'),
            ('phone <number>', 'Test message to phone number'),
            ('guest', 'Test guest notification'),
            ('format <number>', 'Test phone number formatting'),
            ('all', 'Run all tests'),
            ('testall', 'Test message to all staff'),
            ('clear', 'Clear screen'),
            ('quit/exit/q', 'Exit interactive mode'),
        ]
        
        self.stdout.write('\n📖 Available Commands:')
        self.stdout.write('-' * 50)
        for cmd, desc in commands:
            self.stdout.write(f'   {cmd:<15} - {desc}')
        self.stdout.write()
    
    def show_help(self):
        """Show command help"""
        self.stdout.write('\n📖 WhatsApp Test Command Usage:')
        self.stdout.write('=' * 50)
        
        examples = [
            ('Check configuration:', 'python manage.py test_whatsapp --check-api'),
            ('List staff:', 'python manage.py test_whatsapp --list-staff'),
            ('Test staff message:', 'python manage.py test_whatsapp --staff-id 5'),
            ('Test phone message:', 'python manage.py test_whatsapp --phone 6281234567890'),
            ('Test guest notification:', 'python manage.py test_whatsapp --test-guest-notification'),
            ('Run all tests:', 'python manage.py test_whatsapp --all'),
            ('Interactive mode:', 'python manage.py test_whatsapp --interactive'),
            ('Format phone test:', 'python manage.py test_whatsapp --format-phone 081234567890'),
            ('Test all staff:', 'python manage.py test_whatsapp --test-all-staff'),
            ('Dry run mode:', 'python manage.py test_whatsapp --dry-run --all'),
        ]
        
        for desc, cmd in examples:
            self.stdout.write(f'\n{desc}')
            self.stdout.write(f'   {cmd}')
    
    def test_phone_formatting(self, phone_number):
        """Test phone number formatting"""
        self.stdout.write(f'\n📱 Testing Phone Number Formatting')
        self.stdout.write(f'   Input: {phone_number}')
        
        try:
            formatted = whatsapp_service._format_phone_number(phone_number)
            self.stdout.write(f'   Output: {formatted}')
            
            # Validate format
            if formatted.startswith('62') and formatted.isdigit() and len(formatted) >= 11:
                self.stdout.write(self.style.SUCCESS('   ✅ Format valid'))
                return True
            else:
                self.stdout.write(self.style.WARNING('   ⚠️  Format might be invalid'))
                return False
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ❌ Error: {str(e)}'))
            return False
    
    def check_configuration(self):
        """Check WhatsApp configuration"""
        self.stdout.write('\n📋 Checking Configuration...')
        
        # Check settings
        enabled = getattr(settings, 'WHATSAPP_ENABLED', False)
        api_url = getattr(settings, 'WHATSAPP_API_URL', '')
        api_token = getattr(settings, 'WHATSAPP_API_TOKEN', '')
        
        self.stdout.write(f'   WhatsApp Enabled: {self.bool_style(enabled)}')
        self.stdout.write(f'   API URL: {api_url or "Not configured"}')
        self.stdout.write(f'   API Token: {"✅ Configured" if api_token else "❌ Not configured"}')
        
        if api_token and self.verbose:
            self.stdout.write(f'   Token Preview: {api_token[:10]}...')
        
        # Check service initialization
        self.stdout.write(f'   Service Enabled: {self.bool_style(whatsapp_service.enabled)}')
        self.stdout.write(f'   Service Token: {self.bool_style(bool(whatsapp_service.api_token))}')
        self.stdout.write(f'   Service URL: {whatsapp_service.api_url}')
        
        # Check database
        try:
            staff_count = Staff.objects.filter(is_active=True).count()
            whatsapp_enabled_count = Staff.objects.filter(is_active=True, whatsapp_enabled=True).count()
            
            self.stdout.write(f'   Active Staff: {staff_count}')
            self.stdout.write(f'   WhatsApp Enabled Staff: {whatsapp_enabled_count}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   Database Error: {str(e)}'))
            return False
        
        if not enabled or not api_token:
            self.stdout.write(
                self.style.WARNING(
                    '\n⚠️  WhatsApp not properly configured. Please check settings.py'
                )
            )
            return False
        return True
    
    def check_api_status(self):
        """Check Fonnte API status"""
        self.stdout.write('\n🔍 Checking API Status...')
        
        if not whatsapp_service.enabled:
            self.stdout.write(self.style.ERROR('   ❌ WhatsApp service is disabled'))
            return False
        
        if not whatsapp_service.api_token:
            self.stdout.write(self.style.ERROR('   ❌ API token not configured'))
            return False
        
        try:
            # Test basic API connectivity
            headers = {
                'Authorization': whatsapp_service.api_token
            }
            
            self.stdout.write('   🔄 Testing API connection...')
            
            # Check device status
            device_url = 'https://api.fonnte.com/device'
            response = requests.get(device_url, headers=headers, timeout=15)
            
            self.stdout.write(f'   Response Code: {response.status_code}')
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.stdout.write('   ✅ API Connection: Success')
                    
                    if isinstance(data, dict):
                        device_status = data.get('status', 'unknown')
                        device_name = data.get('name', 'Unknown')
                        device_number = data.get('device', 'Unknown')
                        
                        self.stdout.write(f'   📱 Device: {device_name}')
                        self.stdout.write(f'   📞 Number: {device_number}')
                        self.stdout.write(f'   📶 Status: {device_status}')
                        
                        if device_status == 'connect':
                            self.stdout.write(self.style.SUCCESS('   ✅ Device is connected'))
                            return True
                        else:
                            self.stdout.write(self.style.WARNING('   ⚠️  Device is not connected'))
                            return False
                            
                    elif isinstance(data, list):
                        self.stdout.write(f'   📱 Devices: {len(data)} found')
                        connected_count = 0
                        for i, device in enumerate(data[:5]):  # Show first 5 devices
                            name = device.get('name', f'Device {i+1}')
                            status = device.get('status', 'unknown')
                            self.stdout.write(f'     {i+1}. {name}: {status}')
                            if status == 'connect':
                                connected_count += 1
                        
                        if connected_count > 0:
                            self.stdout.write(self.style.SUCCESS(f'   ✅ {connected_count} device(s) connected'))
                            return True
                        else:
                            self.stdout.write(self.style.WARNING('   ⚠️  No devices connected'))
                            return False
                    else:
                        self.stdout.write(f'   📱 Response: {data}')
                        return True
                        
                except json.JSONDecodeError:
                    self.stdout.write(f'   ⚠️  Non-JSON response: {response.text}')
                    return False
                    
            elif response.status_code == 401:
                self.stdout.write(self.style.ERROR('   ❌ Unauthorized: Invalid API token'))
                return False
            elif response.status_code == 403:
                self.stdout.write(self.style.ERROR('   ❌ Forbidden: Check account status'))
                return False
            else:
                self.stdout.write(f'   ❌ API Error: HTTP {response.status_code}')
                self.stdout.write(f'   Response: {response.text}')
                return False
                
        except requests.exceptions.Timeout:
            self.stdout.write(self.style.ERROR('   ❌ API Timeout: Request took too long'))
            return False
        except requests.exceptions.ConnectionError:
            self.stdout.write(self.style.ERROR('   ❌ Connection Error: Cannot reach API'))
            return False
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ❌ Error: {str(e)}'))
            return False
    
    def list_staff(self):
        """List all staff with WhatsApp enabled"""
        self.stdout.write('\n👥 Staff List:')
        
        staff_list = Staff.objects.filter(is_active=True).order_by('department', 'name')
        
        if not staff_list:
            self.stdout.write('   No active staff found')
            return False
        
        whatsapp_enabled = staff_list.filter(whatsapp_enabled=True)
        whatsapp_disabled = staff_list.filter(whatsapp_enabled=False)
        
        if whatsapp_enabled:
            self.stdout.write(f'\n   ✅ WhatsApp Enabled ({whatsapp_enabled.count()}):')
            for staff in whatsapp_enabled:
                status_emoji = {
                    'available': '🟢',
                    'busy': '🟡',
                    'meeting': '🔴',
                    'out': '⚫',
                    'off': '⚪'
                }.get(staff.current_status, '❓')
                
                try:
                    formatted_phone = whatsapp_service._format_phone_number(staff.phone_number)
                except:
                    formatted_phone = staff.phone_number
                
                self.stdout.write(
                    f'     {staff.id:2d}. {staff.name:<30} ({formatted_phone}) '
                    f'{status_emoji} {staff.get_current_status_display()}'
                )
                
                if self.verbose:
                    self.stdout.write(f'         Department: {staff.get_department_display()}')
                    self.stdout.write(f'         Room: {staff.office_room or "Not specified"}')
                    if staff.status_message:
                        self.stdout.write(f'         Message: {staff.status_message}')
        
        if whatsapp_disabled:
            self.stdout.write(f'\n   ❌ WhatsApp Disabled ({whatsapp_disabled.count()}):')
            for staff in whatsapp_disabled:
                self.stdout.write(
                    f'     {staff.id:2d}. {staff.name:<30} ({staff.phone_number})'
                )
        
        return True
    
    def test_staff_message(self, staff_id):
        """Test message to specific staff"""
        self.stdout.write(f'\n📤 Testing message to Staff ID: {staff_id}')
        
        try:
            staff = Staff.objects.get(id=staff_id, is_active=True)
            
            self.stdout.write(f'   Staff: {staff.name}')
            self.stdout.write(f'   Position: {staff.position}')
            self.stdout.write(f'   Department: {staff.get_department_display()}')
            self.stdout.write(f'   Phone: {staff.phone_number}')
            
            try:
                formatted_phone = whatsapp_service._format_phone_number(staff.phone_number)
                self.stdout.write(f'   Formatted: {formatted_phone}')
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'   Phone format error: {str(e)}'))
                return False
            
            self.stdout.write(f'   WhatsApp Enabled: {self.bool_style(staff.whatsapp_enabled)}')
            self.stdout.write(f'   Status: {staff.get_current_status_display()}')
            
            if not staff.whatsapp_enabled:
                self.stdout.write(
                    self.style.WARNING('   ⚠️  WhatsApp disabled for this staff')
                )
                return False
            
            if self.dry_run:
                self.stdout.write(self.style.WARNING('   🏃 DRY RUN: Would send test message here'))
                return True
            
            # Confirm before sending
            if not self.get_confirmation(f'Send test message to {staff.name}?'):
                self.stdout.write('   Cancelled by user')
                return False
            
            # Send test message
            self.stdout.write('   🔄 Sending test message...')
            
            result = whatsapp_service.send_test_message(
                phone_number=staff.phone_number,
                staff_name=staff.name
            )
            
            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS(f'   ✅ Message sent successfully!')
                )
                if 'response_data' in result and self.verbose:
                    self.stdout.write(f'   Response: {result["response_data"]}')
                return True
            else:
                self.stdout.write(
                    self.style.ERROR(f'   ❌ Failed: {result["message"]}')
                )
                if 'error_code' in result:
                    self.stdout.write(f'   Error Code: {result["error_code"]}')
                if 'response_data' in result and self.verbose:
                    self.stdout.write(f'   API Response: {result["response_data"]}')
                return False
                
        except Staff.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'   ❌ Staff with ID {staff_id} not found')
            )
            return False
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'   ❌ Error: {str(e)}')
            )
            if self.verbose:
                self.stdout.write(traceback.format_exc())
            return False
    
    def test_phone_message(self, phone_number):
        """Test message to specific phone number"""
        self.stdout.write(f'\n📤 Testing message to phone: {phone_number}')
        
        try:
            formatted_phone = whatsapp_service._format_phone_number(phone_number)
            self.stdout.write(f'   Original: {phone_number}')
            self.stdout.write(f'   Formatted: {formatted_phone}')
            
            if self.dry_run:
                self.stdout.write(self.style.WARNING('   🏃 DRY RUN: Would send test message here'))
                return True
            
            # Confirm before sending
            if not self.get_confirmation(f'Send test message to {formatted_phone}?'):
                self.stdout.write('   Cancelled by user')
                return False
            
            self.stdout.write('   🔄 Sending test message...')
            
            result = whatsapp_service.send_test_message(
                phone_number=phone_number,
                staff_name="Test User"
            )
            
            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS(f'   ✅ Message sent successfully!')
                )
                if 'response_data' in result and self.verbose:
                    self.stdout.write(f'   Response: {result["response_data"]}')
                return True
            else:
                self.stdout.write(
                    self.style.ERROR(f'   ❌ Failed: {result["message"]}')
                )
                return False
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'   ❌ Error: {str(e)}')
            )
            return False
    
    def test_guest_notification(self):
        """Test guest meeting notification with sample data"""
        self.stdout.write('\n🏢 Testing Guest Meeting Notification...')
        
        try:
            # Get a staff member for testing
            staff = Staff.objects.filter(
                is_active=True, 
                whatsapp_enabled=True
            ).first()
            
            if not staff:
                self.stdout.write(
                    self.style.ERROR('   ❌ No staff found with WhatsApp enabled')
                )
                return False
            
            # Create or get a test guest
            guest, created = Guest.objects.get_or_create(
                name='Test Guest - WhatsApp Test',
                defaults={
                    'email': 'test@example.com',
                    'phone': '081234567890',
                    'company': 'Test Company Pty Ltd'
                }
            )
            
            # Create a test visit log
            visit_log = VisitLog.objects.create(
                guest=guest,
                staff_to_meet=staff,
                visit_purpose='meet_staff',
                visit_description='Testing WhatsApp notification system for guest meeting request. This is a test message to verify the integration is working properly.',
                urgency_level='medium',
                expected_duration=30
            )
            
            self.stdout.write(f'   Guest: {guest.name}')
            self.stdout.write(f'   Company: {guest.company}')
            self.stdout.write(f'   Staff: {staff.name}')
            self.stdout.write(f'   Phone: {staff.phone_number}')
            self.stdout.write(f'   Purpose: {visit_log.get_visit_purpose_display()}')
            self.stdout.write(f'   Urgency: {visit_log.get_urgency_level_display()}')
            self.stdout.write(f'   Duration: {visit_log.expected_duration} minutes')
            
            if self.dry_run:
                self.stdout.write(self.style.WARNING('   🏃 DRY RUN: Would send guest notification here'))
                # Clean up
                visit_log.delete()
                if created:
                    guest.delete()
                return True
            
            # Confirm before sending
            if not self.get_confirmation(f'Send guest notification to {staff.name}?'):
                self.stdout.write('   Cancelled by user')
                # Clean up
                visit_log.delete()
                if created:
                    guest.delete()
                return False
            
            # Send notification
            self.stdout.write('   🔄 Sending guest notification...')
            
            result = whatsapp_service.send_guest_meeting_notification(
                staff=staff,
                guest=guest,
                visit_log=visit_log
            )
            
            success = False
            
            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS(f'   ✅ Notification sent successfully!')
                )
                
                if 'message_id' in result:
                    self.stdout.write(f'   Message ID: {result["message_id"]}')
                    
                # Update visit log
                visit_log.whatsapp_sent = True
                visit_log.whatsapp_sent_at = timezone.now()
                visit_log.save()
                
                self.stdout.write('   ✅ Visit log updated with WhatsApp status')
                success = True
                
            else:
                self.stdout.write(
                    self.style.ERROR(f'   ❌ Failed: {result["message"]}')
                )
                self.stdout.write(f'   Error Code: {result.get("error_code", "Unknown")}')
                
                if 'response_data' in result and self.verbose:
                    self.stdout.write(f'   API Response: {result["response_data"]}')
            
            # Clean up test data
            self.stdout.write('   🧹 Cleaning up test data...')
            visit_log.delete()
            if created:
                guest.delete()
                self.stdout.write('   ✅ Test data cleaned up')
            
            return success
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'   ❌ Error: {str(e)}')
            )
            
            if self.verbose:
                self.stdout.write(traceback.format_exc())
            
            # Clean up on error
            try:
                if 'visit_log' in locals():
                    visit_log.delete()
                if 'created' in locals() and created and 'guest' in locals():
                    guest.delete()
            except:
                pass
            
            return False
    
    def test_all_staff(self):
        """Test message to all WhatsApp-enabled staff"""
        self.stdout.write('\n📤 Testing messages to all WhatsApp-enabled staff...')
        
        staff_list = Staff.objects.filter(
            is_active=True,
            whatsapp_enabled=True
        ).order_by('name')
        
        if not staff_list:
            self.stdout.write(
                self.style.ERROR('   ❌ No staff found with WhatsApp enabled')
            )
            return False
        
        self.stdout.write(f'   Found {staff_list.count()} staff members with WhatsApp enabled')
        
        if self.dry_run:
            self.stdout.write(self.style.WARNING('   🏃 DRY RUN: Would send to all staff'))
            for staff in staff_list:
                self.stdout.write(f'     - {staff.name} ({staff.phone_number})')
            return True
        
        # Show list and confirm
        for staff in staff_list:
            self.stdout.write(f'     - {staff.name} ({staff.phone_number})')
        
        if not self.get_confirmation(f'Send test message to all {staff_list.count()} staff members?'):
            self.stdout.write('   Cancelled by user')
            return False
        
        # Send messages
        success_count = 0
        failed_count = 0
        
        for i, staff in enumerate(staff_list, 1):
            self.stdout.write(f'\n   📤 [{i}/{staff_list.count()}] Sending to {staff.name}...')
            
            try:
                result = whatsapp_service.send_test_message(
                    phone_number=staff.phone_number,
                    staff_name=staff.name
                )
                
                if result['success']:
                    self.stdout.write(self.style.SUCCESS(f'      ✅ Success'))
                    success_count += 1
                else:
                    self.stdout.write(self.style.ERROR(f'      ❌ Failed: {result["message"]}'))
                    failed_count += 1
                
                # Wait between messages to avoid rate limiting
                if i < staff_list.count():  # Don't wait after last message
                    import time
                    time.sleep(3)
                    
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'      ❌ Error: {str(e)}'))
                failed_count += 1
        
        # Summary
        total = success_count + failed_count
        self.stdout.write(f'\n📊 Results: {success_count}/{total} successful, {failed_count} failed')
        
        return success_count > 0
    
    def get_confirmation(self, message):
        """Get user confirmation"""
        try:
            response = input(f'{message} (y/N): ').strip().lower()
            return response in ['y', 'yes']
        except (KeyboardInterrupt, EOFError):
            return False
    
    def bool_style(self, value):
        """Style boolean values"""
        if value:
            return self.style.SUCCESS('✅ Yes')
        else:
            return self.style.ERROR('❌ No')