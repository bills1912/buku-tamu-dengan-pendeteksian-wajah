# Generated after manual database fix
from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('guest_system', '0001_initial'),
    ]

    operations = [
        # Database structure was fixed manually
        # This migration exists only for record keeping
        migrations.RunSQL("SELECT 1;", reverse_sql="SELECT 1;"),
    ]
