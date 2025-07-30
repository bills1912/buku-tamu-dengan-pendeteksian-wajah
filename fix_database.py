#!/usr/bin/env python
"""
Create empty migration for dashboard features
This is just for record keeping after manual database fix
"""

import os
import django
from django.core.management import execute_from_command_line

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'guest_book.settings')
django.setup()

def create_empty_migration():
    """Create empty migration file"""
    migration_content = '''# Generated after manual database fix
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
'''
    
    # Create migrations directory if it doesn't exist
    migrations_dir = 'guest_system/migrations'
    if not os.path.exists(migrations_dir):
        os.makedirs(migrations_dir)
    
    # Write the migration file
    migration_file = os.path.join(migrations_dir, '0002_dashboard_features_manual.py')
    with open(migration_file, 'w', encoding='utf8') as f:
        f.write(migration_content)
    
    print(f"✅ Created empty migration: {migration_file}")
    print("Now you can run: python manage.py migrate")

if __name__ == "__main__":
    create_empty_migration()