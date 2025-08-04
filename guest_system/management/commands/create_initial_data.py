# management/commands/create_initial_data.py

from django.core.management.base import BaseCommand
from django.utils import timezone
from guest_system.models import Version, ChangelogItem, Staff
import json

class Command(BaseCommand):
    help = 'Create initial data for Smart Guest Book System'

    def handle(self, *args, **options):
        self.stdout.write('Creating initial data...')
        
        # Create version data
        self.create_version_data()
        
        # Create staff data
        self.create_staff_data()
        
        self.stdout.write(
            self.style.SUCCESS('Successfully created initial data!')
        )

    def create_version_data(self):
        """Create initial version and changelog data"""
        
        # Version 2.0.0 - Current with new features
        version_200, created = Version.objects.get_or_create(
            version_number='2.0.0',
            defaults={
                'title': 'Major Update - Staff Selection & WhatsApp Integration',
                'description': 'Pembaruan besar dengan fitur pemilihan staff dan integrasi WhatsApp untuk notifikasi real-time.',
                'is_current': True,
                'release_date': timezone.now()
            }
        )
        
        if created:
            # Changelog items for v2.0.0
            changelog_items_200 = [
                {
                    'item_type': 'new',
                    'title': 'Fitur Pemilihan Staff',
                    'description': 'Tamu dapat memilih staff spesifik yang ingin ditemui dengan interface yang intuitif dan estetik.',
                    'is_highlighted': True
                },
                {
                    'item_type': 'new',
                    'title': 'Integrasi WhatsApp Notification',
                    'description': 'Sistem otomatis mengirim notifikasi WhatsApp ke staff ketika ada tamu yang ingin bertemu.',
                    'is_highlighted': True
                },
                {
                    'item_type': 'new',
                    'title': 'Changelog & Version History',
                    'description': 'Tracking lengkap semua pembaruan sistem dengan interface yang profesional.',
                    'is_highlighted': True
                },
                {
                    'item_type': 'new',
                    'title': 'Enhanced Staff Management',
                    'description': 'Manajemen staff dengan status real-time, foto, dan informasi kontak lengkap.',
                    'is_highlighted': False
                },
                {
                    'item_type': 'improvement',
                    'title': 'UI/UX Enhancement',
                    'description': 'Peningkatan tampilan interface dengan animasi smooth dan responsive design.',
                    'is_highlighted': False
                },
                {
                    'item_type': 'improvement',
                    'title': 'Enhanced Guest Book Form',
                    'description': 'Form buku tamu yang lebih komprehensif dengan custom select components.',
                    'is_highlighted': False
                },
                {
                    'item_type': 'improvement',
                    'title': 'Better Dashboard Analytics',
                    'description': 'Dashboard dengan informasi staff meeting dan status WhatsApp notification.',
                    'is_highlighted': False
                },
                {
                    'item_type': 'fix',
                    'title': 'Navigation Flow Optimization',
                    'description': 'Optimasi alur navigasi dari gesture recognition ke staff selection.',
                    'is_highlighted': False
                }
            ]
            
            for item_data in changelog_items_200:
                ChangelogItem.objects.create(
                    version=version_200,
                    **item_data
                )
        
        # Version 1.5.0 - Previous version
        version_150, created = Version.objects.get_or_create(
            version_number='1.5.0',
            defaults={
                'title': 'Dashboard & Monitoring Enhancement',
                'description': 'Peningkatan sistem monitoring dengan dashboard real-time dan notifikasi.',
                'is_current': False,
                'release_date': timezone.now() - timezone.timedelta(days=30)
            }
        )
        
        if created:
            changelog_items_150 = [
                {
                    'item_type': 'new',
                    'title': 'Real-time Dashboard',
                    'description': 'Dashboard monitoring dengan statistik live dan visualisasi data.',
                    'is_highlighted': True
                },
                {
                    'item_type': 'new',
                    'title': 'Notification System',
                    'description': 'Sistem notifikasi untuk tracking aktivitas tamu dan reminder.',
                    'is_highlighted': True
                },
                {
                    'item_type': 'improvement',
                    'title': 'Enhanced Analytics',
                    'description': 'Grafik dan chart untuk analisis trend kunjungan harian.',
                    'is_highlighted': False
                },
                {
                    'item_type': 'improvement',
                    'title': 'Mobile Responsiveness',
                    'description': 'Optimasi tampilan untuk device mobile dan tablet.',
                    'is_highlighted': False
                },
                {
                    'item_type': 'fix',
                    'title': 'Performance Optimization',
                    'description': 'Optimasi performa loading dan response time sistem.',
                    'is_highlighted': False
                }
            ]
            
            for item_data in changelog_items_150:
                ChangelogItem.objects.create(
                    version=version_150,
                    **item_data
                )
        
        # Version 1.0.0 - Initial version
        version_100, created = Version.objects.get_or_create(
            version_number='1.0.0',
            defaults={
                'title': 'Initial Release - Smart Guest Book',
                'description': 'Rilis perdana sistem buku tamu dengan AI face recognition dan gesture detection.',
                'is_current': False,
                'release_date': timezone.now() - timezone.timedelta(days=90)
            }
        )
        
        if created:
            changelog_items_100 = [
                {
                    'item_type': 'new',
                    'title': 'Face Recognition System',
                    'description': 'Sistem pengenalan wajah otomatis untuk identifikasi tamu.',
                    'is_highlighted': True
                },
                {
                    'item_type': 'new',
                    'title': 'Gesture Detection',
                    'description': 'Deteksi gestur tangan untuk pemilihan layanan secara intuitif.',
                    'is_highlighted': True
                },
                {
                    'item_type': 'new',
                    'title': 'Digital Guest Book',
                    'description': 'Sistem buku tamu digital dengan form interaktif.',
                    'is_highlighted': True
                },
                {
                    'item_type': 'new',
                    'title': 'Audio Guidance',
                    'description': 'Panduan suara untuk membantu tamu dalam menggunakan sistem.',
                    'is_highlighted': False
                },
                {
                    'item_type': 'new',
                    'title': 'Basic Admin Panel',
                    'description': 'Panel administrasi untuk manajemen data tamu dan kunjungan.',
                    'is_highlighted': False
                }
            ]
            
            for item_data in changelog_items_100:
                ChangelogItem.objects.create(
                    version=version_100,
                    **item_data
                )

    def create_staff_data(self):
        """Create initial staff data"""
        
        staff_data = [
            {
                'name': 'Dr. Ahmad Suryana, S.Si, M.Stat',
                'position': 'Kepala Kantor',
                'department': 'kepala_kantor',
                'phone_number': '628123456789',
                'email': 'kepala@bps-sijunjung.go.id',
                'office_room': 'Ruang Kepala Kantor',
                'current_status': 'available',
                'whatsapp_enabled': True
            },
            {
                'name': 'Siti Aminah, S.ST',
                'position': 'Kepala Subbag Umum & Kepegawaian',
                'department': 'subbag_umum',
                'phone_number': '628234567890',
                'email': 'umum@bps-sijunjung.go.id',
                'office_room': 'Ruang Subbag Umum',
                'current_status': 'available',
                'whatsapp_enabled': True
            },
            {
                'name': 'Budi Santoso, S.Si',
                'position': 'Koordinator Fungsi PST',
                'department': 'pst',
                'phone_number': '628345678901',
                'email': 'pst@bps-sijunjung.go.id',
                'office_room': 'Ruang PST',
                'current_status': 'available',
                'whatsapp_enabled': True
            },
            {
                'name': 'Rina Sari, S.ST',
                'position': 'Koordinator Fungsi IPDS',
                'department': 'ipds',
                'phone_number': '628456789012',
                'email': 'ipds@bps-sijunjung.go.id',
                'office_room': 'Ruang IPDS',
                'current_status': 'available',
                'whatsapp_enabled': True
            },
            {
                'name': 'Joko Widodo, S.ST',
                'position': 'Koordinator Fungsi Statistik Distribusi',
                'department': 'distribusi',
                'phone_number': '628567890123',
                'email': 'distribusi@bps-sijunjung.go.id',
                'office_room': 'Ruang Statistik Distribusi',
                'current_status': 'available',
                'whatsapp_enabled': True
            },
            {
                'name': 'Maya Sari, S.Si',
                'position': 'Koordinator Fungsi Statistik Produksi',
                'department': 'produksi',
                'phone_number': '628678901234',
                'email': 'produksi@bps-sijunjung.go.id',
                'office_room': 'Ruang Statistik Produksi',
                'current_status': 'busy',
                'status_message': 'Sedang rapat koordinasi',
                'whatsapp_enabled': True
            },
            {
                'name': 'Dewi Lestari, S.ST',
                'position': 'Koordinator Fungsi Statistik Sosial',
                'department': 'sosial',
                'phone_number': '628789012345',
                'email': 'sosial@bps-sijunjung.go.id',
                'office_room': 'Ruang Statistik Sosial',
                'current_status': 'available',
                'whatsapp_enabled': True
            },
            {
                'name': 'Rahmat Hidayat, S.Si',
                'position': 'Koordinator Fungsi Neraca & Analisis',
                'department': 'neraca',
                'phone_number': '628890123456',
                'email': 'neraca@bps-sijunjung.go.id',
                'office_room': 'Ruang Neraca Wilayah',
                'current_status': 'meeting',
                'status_message': 'Rapat evaluasi triwulan',
                'whatsapp_enabled': True
            }
        ]
        
        for staff_info in staff_data:
            staff, created = Staff.objects.get_or_create(
                name=staff_info['name'],
                defaults=staff_info
            )
            if created:
                self.stdout.write(f'Created staff: {staff.name}')
            else:
                self.stdout.write(f'Staff already exists: {staff.name}')

        self.stdout.write(f'Staff data creation completed!')


# Run this command with: python manage.py create_initial_data