from django.db import models
from django.utils import timezone
import json

class Version(models.Model):
    """Model for tracking application versions and changelogs"""
    version_number = models.CharField(max_length=20, unique=True)
    release_date = models.DateTimeField(default=timezone.now)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    is_current = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-release_date']
    
    def save(self, *args, **kwargs):
        if self.is_current:
            # Set all other versions to not current
            Version.objects.exclude(pk=self.pk).update(is_current=False)
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Version {self.version_number}"

class ChangelogItem(models.Model):
    """Model for individual changelog items"""
    ITEM_TYPES = [
        ('new', 'New Feature'),
        ('improvement', 'Improvement'), 
        ('fix', 'Bug Fix'),
        ('security', 'Security Update'),
        ('breaking', 'Breaking Change'),
    ]
    
    version = models.ForeignKey(Version, on_delete=models.CASCADE, related_name='changelog_items')
    item_type = models.CharField(max_length=20, choices=ITEM_TYPES, default='improvement')
    title = models.CharField(max_length=200)
    description = models.TextField()
    is_highlighted = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['item_type', 'created_at']
    
    @property
    def type_icon(self):
        icons = {
            'new': 'fas fa-plus-circle',
            'improvement': 'fas fa-arrow-up',
            'fix': 'fas fa-wrench',
            'security': 'fas fa-shield-alt',
            'breaking': 'fas fa-exclamation-triangle'
        }
        return icons.get(self.item_type, 'fas fa-circle')
    
    @property
    def type_color(self):
        colors = {
            'new': '#10b981',
            'improvement': '#3b82f6', 
            'fix': '#f59e0b',
            'security': '#ef4444',
            'breaking': '#8b5cf6'
        }
        return colors.get(self.item_type, '#6b7280')
    
    def __str__(self):
        return f"{self.version.version_number} - {self.title}"

class Staff(models.Model):
    """Model for staff members who can be visited"""
    DEPARTMENTS = [
        ('kepala_kantor', 'Kepala Kantor'),
        ('sekretaris', 'Sekretaris'),
        ('subbag_umum', 'Subbag Umum & Kepegawaian'),
        ('pst', 'Pelayanan Statistik Terpadu'),
        ('ipds', 'Integrasi Pengolahan dan Diseminasi Statistik'),
        ('distribusi', 'Statistik Distribusi'),
        ('produksi', 'Statistik Produksi'),
        ('sosial', 'Statistik Sosial'),
        ('neraca', 'Neraca Wilayah dan Analisis Statistik'),
    ]
    
    STATUS_CHOICES = [
        ('available', 'Tersedia'),
        ('busy', 'Sedang Sibuk'), 
        ('meeting', 'Sedang Rapat'),
        ('out', 'Sedang Keluar'),
        ('off', 'Tidak Masuk'),
    ]
    
    name = models.CharField(max_length=200)
    position = models.CharField(max_length=200)
    department = models.CharField(max_length=50, choices=DEPARTMENTS)
    phone_number = models.CharField(max_length=20, help_text="Format: 628xxxxxxxxxx (untuk WhatsApp)")
    email = models.EmailField(blank=True, null=True)
    office_room = models.CharField(max_length=100, blank=True, null=True)
    photo = models.ImageField(upload_to='staff_photos/', blank=True, null=True)
    current_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='available')
    status_message = models.CharField(max_length=200, blank=True, null=True, help_text="Pesan status tambahan")
    whatsapp_enabled = models.BooleanField(default=True, help_text="Aktifkan notifikasi WhatsApp")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['department', 'name']
        verbose_name = 'Staff'
        verbose_name_plural = 'Staff'
    
    @property
    def status_color(self):
        colors = {
            'available': '#10b981',
            'busy': '#f59e0b', 
            'meeting': '#ef4444',
            'out': '#6b7280',
            'off': '#9ca3af'
        }
        return colors.get(self.current_status, '#6b7280')
    
    @property
    def status_icon(self):
        icons = {
            'available': 'fas fa-check-circle',
            'busy': 'fas fa-clock',
            'meeting': 'fas fa-users',
            'out': 'fas fa-door-open',
            'off': 'fas fa-times-circle'
        }
        return icons.get(self.current_status, 'fas fa-circle')
    
    @property
    def department_display(self):
        return dict(self.DEPARTMENTS).get(self.department, self.department)
    
    @property
    def whatsapp_number(self):
        """Format phone number for WhatsApp"""
        if self.phone_number.startswith('08'):
            return '628' + self.phone_number[2:]
        elif self.phone_number.startswith('+62'):
            return self.phone_number[1:]
        return self.phone_number
    
    def __str__(self):
        return f"{self.name} - {self.position}"

class Guest(models.Model):
    VISIT_PURPOSE_CHOICES = [
        ('data_service', 'Pelayanan Data'),
        ('meet_staff', 'Menemui Pegawai/Kepala Kantor'),
        ('other_activity', 'Mengikuti Kegiatan Lainnya'),
    ]
    
    name = models.CharField(max_length=200)
    email = models.EmailField(blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    company = models.CharField(max_length=200, blank=True, null=True)
    face_encoding = models.TextField()  # JSON string of face encoding
    face_image = models.ImageField(upload_to='face_images/', null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def set_face_encoding(self, encoding_array):
        self.face_encoding = json.dumps(encoding_array.tolist())
    
    def get_face_encoding(self):
        return json.loads(self.face_encoding)
    
    @property
    def status(self):
        """Get current guest status"""
        latest_visit = self.visitlog_set.filter(check_out_time__isnull=True).first()
        if latest_visit:
            if latest_visit.visit_description:
                return 'active_visit'
            else:
                return 'pending_guestbook'
        return 'registered'
    
    @property
    def current_visit(self):
        """Get current active visit"""
        return self.visitlog_set.filter(check_out_time__isnull=True).first()
    
    def __str__(self):
        return self.name

class VisitLog(models.Model):
    URGENCY_CHOICES = [
        ('low', 'Rendah'),
        ('medium', 'Sedang'),
        ('high', 'Tinggi'),
    ]
    
    guest = models.ForeignKey(Guest, on_delete=models.CASCADE)
    visit_purpose = models.CharField(max_length=20, choices=Guest.VISIT_PURPOSE_CHOICES)
    visit_description = models.TextField(blank=True, null=True)
    urgency_level = models.CharField(max_length=10, choices=URGENCY_CHOICES, default='low')
    expected_duration = models.IntegerField(blank=True, null=True)  # in minutes
    checkout_notes = models.TextField(blank=True, null=True)
    visit_date = models.DateTimeField(default=timezone.now)
    check_in_time = models.DateTimeField(auto_now_add=True)
    check_out_time = models.DateTimeField(blank=True, null=True)
    
    # New fields for staff meeting
    staff_to_meet = models.ForeignKey(Staff, on_delete=models.SET_NULL, blank=True, null=True, help_text="Staff yang ingin ditemui")
    whatsapp_sent = models.BooleanField(default=False, help_text="Status pengiriman WhatsApp")
    whatsapp_sent_at = models.DateTimeField(blank=True, null=True)
    
    @property
    def duration_minutes(self):
        """Get visit duration in minutes"""
        end_time = self.check_out_time or timezone.now()
        duration = end_time - self.check_in_time
        return int(duration.total_seconds() / 60)
    
    @property
    def is_overdue(self):
        """Check if visit is overdue based on expected duration"""
        if not self.expected_duration:
            return False
        return self.duration_minutes > self.expected_duration
    
    def __str__(self):
        return f"{self.guest.name} - {self.visit_date.strftime('%Y-%m-%d %H:%M')}"

class Notification(models.Model):
    NOTIFICATION_TYPES = [
        ('new_registration', 'New Registration'),
        ('guestbook_reminder', 'Guestbook Reminder'),
        ('guestbook_completed', 'Guestbook Completed'),
        ('guest_checkout', 'Guest Checkout'),
        ('long_visit', 'Long Visit Warning'),
        ('staff_meeting_request', 'Staff Meeting Request'),
        ('whatsapp_sent', 'WhatsApp Notification Sent'),
    ]
    
    guest = models.ForeignKey(Guest, on_delete=models.CASCADE)
    notification_type = models.CharField(max_length=50, choices=NOTIFICATION_TYPES)
    title = models.CharField(max_length=200)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Optional reference to staff for staff-related notifications
    related_staff = models.ForeignKey(Staff, on_delete=models.SET_NULL, blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
    
    @property
    def time_ago(self):
        """Get human-readable time difference"""
        now = timezone.now()
        diff = now - self.created_at
        
        if diff.days > 0:
            return f"{diff.days} hari yang lalu"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} jam yang lalu"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} menit yang lalu"
        else:
            return "Baru saja"
    
    def __str__(self):
        return f"{self.title} - {self.guest.name}"