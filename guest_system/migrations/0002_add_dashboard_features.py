# Fixed migration file
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone

class Migration(migrations.Migration):

    dependencies = [
        ('guest_system', '0001_initial'),
    ]

    operations = [
        # Add new fields to VisitLog (skip check_out_time if it already exists)
        migrations.AddField(
            model_name='visitlog',
            name='urgency_level',
            field=models.CharField(
                choices=[('low', 'Rendah'), ('medium', 'Sedang'), ('high', 'Tinggi')], 
                default='low', 
                max_length=10
            ),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='visitlog',
            name='expected_duration',
            field=models.IntegerField(blank=True, null=True),
        ),
        
        # Modify visit_description to allow blank/null (might already be like this)
        migrations.AlterField(
            model_name='visitlog',
            name='visit_description',
            field=models.TextField(blank=True, null=True),
        ),
        
        # Create Notification model
        migrations.CreateModel(
            name='Notification',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('notification_type', models.CharField(
                    choices=[
                        ('new_registration', 'New Registration'), 
                        ('guestbook_reminder', 'Guestbook Reminder'), 
                        ('guestbook_completed', 'Guestbook Completed'), 
                        ('guest_checkout', 'Guest Checkout'), 
                        ('long_visit', 'Long Visit Warning')
                    ], 
                    max_length=50
                )),
                ('title', models.CharField(max_length=200)),
                ('message', models.TextField()),
                ('is_read', models.BooleanField(default=False)),
                ('is_resolved', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('guest', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='guest_system.guest')),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
    ]