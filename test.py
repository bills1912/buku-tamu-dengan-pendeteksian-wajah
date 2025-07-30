# ==============================================================================
# COMPREHENSIVE DJANGO DEBUG & FIX TOOL
# ==============================================================================

import os
import sys
import django
from pathlib import Path

def setup_django():
    """Setup Django environment"""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'guest_book.settings')
    try:
        django.setup()
        return True
    except Exception as e:
        print(f"❌ Django setup failed: {e}")
        return False

def check_project_structure():
    """Check basic project structure"""
    print("🔍 CHECKING PROJECT STRUCTURE")
    print("=" * 50)
    
    required_files = {
        'manage.py': 'Django management script',
        'guest_book/settings.py': 'Django settings',
        'guest_book/urls.py': 'Main URL configuration',
        'guest_system/__init__.py': 'App package marker',
        'guest_system/models.py': 'Database models',
        'guest_system/apps.py': 'App configuration',
        'guest_system/admin.py': 'Admin configuration',
        'guest_system/views.py': 'View functions',
    }
    
    all_exists = True
    for file_path, description in required_files.items():
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"{status} {file_path:<30} - {description}")
        if not exists:
            all_exists = False
    
    return all_exists

def check_settings():
    """Check Django settings configuration"""
    print("\n🔍 CHECKING DJANGO SETTINGS")
    print("=" * 50)
    
    try:
        from django.conf import settings
        
        # Check INSTALLED_APPS
        installed_apps = getattr(settings, 'INSTALLED_APPS', [])
        print(f"INSTALLED_APPS: {len(installed_apps)} apps")
        
        guest_system_installed = 'guest_system' in installed_apps
        status = "✅" if guest_system_installed else "❌"
        print(f"{status} guest_system in INSTALLED_APPS: {guest_system_installed}")
        
        if not guest_system_installed:
            print("📝 INSTALLED_APPS content:")
            for i, app in enumerate(installed_apps, 1):
                print(f"   {i}. {app}")
        
        # Check DATABASE
        databases = getattr(settings, 'DATABASES', {})
        default_db = databases.get('default', {})
        db_engine = default_db.get('ENGINE', 'Not configured')
        db_name = default_db.get('NAME', 'Not configured')
        
        print(f"✅ Database engine: {db_engine}")
        print(f"✅ Database name: {db_name}")
        
        return guest_system_installed
        
    except Exception as e:
        print(f"❌ Settings check failed: {e}")
        return False

def check_models():
    """Check if models are properly defined"""
    print("\n🔍 CHECKING MODELS")
    print("=" * 50)
    
    try:
        # Try to import models
        from guest_system.models import Guest, VisitLog
        print("✅ Models imported successfully")
        
        # Check model fields
        guest_fields = [f.name for f in Guest._meta.get_fields()]
        visit_fields = [f.name for f in VisitLog._meta.get_fields()]
        
        print(f"✅ Guest model fields ({len(guest_fields)}): {guest_fields}")
        print(f"✅ VisitLog model fields ({len(visit_fields)}): {visit_fields}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False

def check_app_config():
    """Check app configuration"""
    print("\n🔍 CHECKING APP CONFIGURATION")
    print("=" * 50)
    
    try:
        from django.apps import apps
        
        # Get all app configs
        app_configs = apps.get_app_configs()
        app_names = [app.name for app in app_configs]
        
        print(f"📋 Loaded apps ({len(app_names)}):")
        for app in app_configs:
            marker = "🎯" if app.name == 'guest_system' else "  "
            print(f"{marker} {app.name} - {app.verbose_name}")
        
        # Check guest_system specifically
        if 'guest_system' in app_names:
            guest_app = apps.get_app_config('guest_system')
            models = guest_app.get_models()
            print(f"\n✅ guest_system app loaded successfully")
            print(f"✅ Models in guest_system: {[model.__name__ for model in models]}")
            return True
        else:
            print("\n❌ guest_system app not loaded")
            return False
            
    except Exception as e:
        print(f"❌ App config check failed: {e}")
        return False

def check_migrations():
    """Check migration status"""
    print("\n🔍 CHECKING MIGRATIONS")
    print("=" * 50)
    
    try:
        from django.core.management import execute_from_command_line
        from django.db.migrations.executor import MigrationExecutor
        from django.db import connection
        
        # Check migrations directory
        migrations_dir = Path('guest_system/migrations')
        if migrations_dir.exists():
            migration_files = list(migrations_dir.glob('*.py'))
            migration_files = [f for f in migration_files if f.name != '__init__.py']
            
            print(f"📁 Migrations directory exists")
            print(f"📄 Migration files found: {len(migration_files)}")
            for f in migration_files:
                print(f"   - {f.name}")
        else:
            print("❌ Migrations directory not found")
            return False
        
        # Check migration state in database
        executor = MigrationExecutor(connection)
        plan = executor.migration_plan(executor.loader.graph.leaf_nodes())
        
        print(f"\n📊 Migration plan: {len(plan)} pending migrations")
        if plan:
            for migration, backwards in plan:
                direction = "⬅️ ROLLBACK" if backwards else "➡️ APPLY"
                print(f"   {direction} {migration}")
        else:
            print("✅ No pending migrations")
        
        # Check applied migrations
        applied = executor.loader.applied_migrations
        guest_migrations = [m for m in applied if m[0] == 'guest_system']
        print(f"✅ Applied guest_system migrations: {len(guest_migrations)}")
        for app, name in guest_migrations:
            print(f"   - {app}.{name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration check failed: {e}")
        return False

def check_database():
    """Check database tables"""
    print("\n🔍 CHECKING DATABASE TABLES")
    print("=" * 50)
    
    try:
        from django.db import connection
        
        with connection.cursor() as cursor:
            # Check all tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            all_tables = cursor.fetchall()
            
            print(f"📊 Total database tables: {len(all_tables)}")
            
            # Check guest_system tables specifically
            guest_tables = [table[0] for table in all_tables if 'guest_system' in table[0]]
            
            if guest_tables:
                print(f"✅ guest_system tables found: {len(guest_tables)}")
                for table in guest_tables:
                    print(f"   - {table}")
                    
                    # Check table structure
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    print(f"     📋 Columns: {len(columns)}")
                    
                return True
            else:
                print("❌ No guest_system tables found")
                print("📋 Available tables:")
                for table in all_tables:
                    print(f"   - {table[0]}")
                return False
                
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def create_minimal_models():
    """Create minimal models file"""
    print("\n🔧 CREATING MINIMAL MODELS")
    print("=" * 50)
    
    models_content = '''from django.db import models
from django.utils import timezone
import json

class Guest(models.Model):
    name = models.CharField(max_length=200)
    email = models.EmailField(blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    company = models.CharField(max_length=200, blank=True, null=True)
    face_encoding = models.TextField()
    face_image = models.ImageField(upload_to='face_images/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Tamu"
        verbose_name_plural = "Data Tamu"
    
    def __str__(self):
        return self.name

class VisitLog(models.Model):
    VISIT_PURPOSE_CHOICES = [
        ('data_service', 'Pelayanan Data'),
        ('meet_staff', 'Menemui Pegawai'),
    ]
    
    guest = models.ForeignKey(Guest, on_delete=models.CASCADE)
    visit_purpose = models.CharField(max_length=20, choices=VISIT_PURPOSE_CHOICES)
    visit_description = models.TextField()
    visit_date = models.DateTimeField(default=timezone.now)
    check_in_time = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Kunjungan"
        verbose_name_plural = "Data Kunjungan"
    
    def __str__(self):
        return f"{self.guest.name} - {self.visit_date}"
'''
    
    try:
        with open('guest_system/models.py', 'w', encoding='utf-8') as f:
            f.write(models_content)
        print("✅ Minimal models.py created")
        return True
    except Exception as e:
        print(f"❌ Failed to create models.py: {e}")
        return False

def create_minimal_admin():
    """Create minimal admin file"""
    admin_content = '''from django.contrib import admin
from .models import Guest, VisitLog

@admin.register(Guest)
class GuestAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'email']

@admin.register(VisitLog)
class VisitLogAdmin(admin.ModelAdmin):
    list_display = ['guest', 'visit_purpose', 'visit_date']
    list_filter = ['visit_purpose', 'visit_date']
    search_fields = ['guest__name']
'''
    
    try:
        with open('guest_system/admin.py', 'w', encoding='utf-8') as f:
            f.write(admin_content)
        print("✅ Minimal admin.py created")
        return True
    except Exception as e:
        print(f"❌ Failed to create admin.py: {e}")
        return False

def fix_apps_config():
    """Fix apps.py configuration"""
    apps_content = '''from django.apps import AppConfig

class GuestSystemConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'guest_system'
    verbose_name = 'Guest System'
'''
    
    init_content = '''default_app_config = 'guest_system.apps.GuestSystemConfig'
'''
    
    try:
        with open('guest_system/apps.py', 'w', encoding='utf-8') as f:
            f.write(apps_content)
        
        with open('guest_system/__init__.py', 'w', encoding='utf-8') as f:
            f.write(init_content)
        
        print("✅ Apps configuration fixed")
        return True
    except Exception as e:
        print(f"❌ Failed to fix apps config: {e}")
        return False

def force_makemigrations():
    """Force create migrations"""
    print("\n🔧 FORCING MIGRATION CREATION")
    print("=" * 50)
    
    try:
        # Remove existing migrations
        migrations_dir = Path('guest_system/migrations')
        if migrations_dir.exists():
            for f in migrations_dir.glob('*.py'):
                if f.name != '__init__.py':
                    f.unlink()
                    print(f"🗑️ Removed {f.name}")
        else:
            migrations_dir.mkdir(exist_ok=True)
        
        # Ensure __init__.py exists
        init_file = migrations_dir / '__init__.py'
        if not init_file.exists():
            init_file.write_text('')
        
        # Run makemigrations
        from django.core.management import call_command
        
        print("📝 Creating initial migration...")
        call_command('makemigrations', 'guest_system', verbosity=2)
        
        print("📝 Applying migrations...")
        call_command('migrate', verbosity=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Force migration failed: {e}")
        return False

def run_comprehensive_fix():
    """Run comprehensive fix process"""
    print("🔧 COMPREHENSIVE DJANGO FIX")
    print("=" * 80)
    
    # Step 1: Check project structure
    if not check_project_structure():
        print("\n❌ Project structure issues found. Please fix manually.")
        return False
    
    # Step 2: Fix basic files
    print("\n🔧 Fixing basic configuration files...")
    fix_apps_config()
    create_minimal_models()
    create_minimal_admin()
    
    # Step 3: Setup Django
    if not setup_django():
        print("\n❌ Django setup failed. Check your settings.py")
        return False
    
    # Step 4: Check settings
    if not check_settings():
        print("\n❌ Settings configuration issues. Check INSTALLED_APPS.")
        return False
    
    # Step 5: Check models
    if not check_models():
        print("\n❌ Model issues found.")
        return False
    
    # Step 6: Check app config
    if not check_app_config():
        print("\n❌ App configuration issues.")
        return False
    
    # Step 7: Force migrations
    if not force_makemigrations():
        print("\n❌ Migration creation failed.")
        return False
    
    # Step 8: Final verification
    print("\n🔍 FINAL VERIFICATION")
    print("=" * 50)
    
    success = (
        check_migrations() and
        check_database()
    )
    
    if success:
        print("\n🎉 SUCCESS! Django is properly configured and migrated.")
        print("\nNext steps:")
        print("1. python manage.py createsuperuser")
        print("2. python manage.py runserver")
        print("3. Visit http://127.0.0.1:8000/admin/")
    else:
        print("\n❌ Some issues remain. Check the output above.")
    
    return success

if __name__ == "__main__":
    # Suppress pygame
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    
    run_comprehensive_fix()