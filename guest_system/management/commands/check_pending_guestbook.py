from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from guest_system.models import Guest, Notification

class Command(BaseCommand):
    help = 'Check for guests who haven\'t filled guestbook and create notifications'

    def add_arguments(self, parser):
        parser.add_argument(
            '--minutes',
            type=int,
            default=10,
            help='Minutes after registration to check for pending guestbook (default: 10)'
        )

    def handle(self, *args, **options):
        minutes = options['minutes']
        cutoff_time = timezone.now() - timedelta(minutes=minutes)
        
        # Find guests registered more than X minutes ago but haven't filled guestbook
        pending_guests = Guest.objects.filter(
            created_at__lt=cutoff_time,
            visitlog__isnull=True
        ).exclude(
            notification__notification_type='guestbook_reminder',
            notification__created_at__gt=timezone.now() - timedelta(hours=1)
        )
        
        notifications_created = 0
        
        for guest in pending_guests:
            elapsed_minutes = int((timezone.now() - guest.created_at).total_seconds() / 60)
            
            # Create notification
            Notification.objects.create(
                guest=guest,
                notification_type='guestbook_reminder',
                title='Belum Mengisi Buku Tamu',
                message=f'{guest.name} telah terdaftar namun belum mengisi buku tamu selama {elapsed_minutes} menit.'
            )
            
            notifications_created += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f'Created reminder notification for {guest.name} (registered {elapsed_minutes} minutes ago)'
                )
            )
        
        if notifications_created == 0:
            self.stdout.write(
                self.style.SUCCESS('No pending guestbook notifications needed')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully created {notifications_created} reminder notifications'
                )
            )