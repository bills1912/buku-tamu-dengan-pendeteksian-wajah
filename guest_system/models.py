from django.db import models
from django.utils import timezone
import json

class Guest(models.Model):
    VISIT_PURPOSE_CHOICES = [
        ('data_service', 'Pelayanan Data'),
        ('meet_staff', 'Menemui Pegawai/Kepala Kantor'),
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
    ]
    
    guest = models.ForeignKey(Guest, on_delete=models.CASCADE)
    notification_type = models.CharField(max_length=50, choices=NOTIFICATION_TYPES)
    title = models.CharField(max_length=200)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
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