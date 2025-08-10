#!/usr/bin/env python3
"""
Script untuk mengatasi NumPy compatibility issues
Solusi untuk: ValueError: numpy.dtype size changed, may indicate binary incompatibility
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_current_versions():
    """Check versi library saat ini"""
    logger.info("=== CHECKING CURRENT VERSIONS ===")
    
    packages_to_check = [
        'numpy', 'opencv-python', 'tensorflow', 'deepface', 
        'scikit-learn', 'pandas', 'Pillow', 'scipy'
    ]
    
    current_versions = {}
    
    for package in packages_to_check:
        try:
            result = subprocess.run([sys.executable, '-c', f'import {package.replace("-", "_")}; print({package.replace("-", "_")}.__version__)'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                current_versions[package] = version
                logger.info(f"{package}: {version}")
            else:
                current_versions[package] = "Not installed"
                logger.warning(f"{package}: Not installed")
        except Exception as e:
            current_versions[package] = f"Error: {e}"
            logger.error(f"{package}: Error checking version")
    
    return current_versions

def uninstall_problematic_packages():
    """Uninstall packages yang mungkin bermasalah"""
    logger.info("=== UNINSTALLING PROBLEMATIC PACKAGES ===")
    
    packages_to_uninstall = [
        'numpy', 'opencv-python', 'opencv-contrib-python', 
        'scikit-learn', 'scipy', 'pandas', 'tensorflow',
        'deepface', 'Pillow'
    ]
    
    for package in packages_to_uninstall:
        try:
            logger.info(f"Uninstalling {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', package, '-y'])
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to uninstall {package} (might not be installed)")
        except Exception as e:
            logger.error(f"Error uninstalling {package}: {e}")

def install_compatible_versions():
    """Install versi yang kompatibel dengan Python 3.9.23"""
    logger.info("=== INSTALLING COMPATIBLE VERSIONS ===")
    
    # Urutan instalasi penting - NumPy dulu, lalu yang lain
    install_sequence = [
        # NumPy versi yang lebih kompatibel
        "numpy==1.24.3",  # Versi yang lebih stable untuk Python 3.9
        
        # Core scientific libraries
        "scipy==1.10.1",
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        
        # Image processing
        "Pillow==10.0.1",
        "opencv-python==4.8.0.76",  # Versi yang kompatibel dengan NumPy 1.24
        
        # Machine Learning
        "tensorflow==2.13.1",  # Versi yang lebih kompatibel dengan Python 3.9
        
        # DeepFace dan dependencies
        "deepface==0.0.79",  # Versi yang lebih stable
        "mtcnn==0.1.1",
        "retina-face==0.0.13",
        
        # Utilities
        "tqdm==4.66.1",
        "gdown==4.7.1",
        "requests==2.31.0"
    ]
    
    for package in install_sequence:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                package, '--no-cache-dir', '--force-reinstall'
            ])
            logger.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error installing {package}: {e}")
            return False
    
    return True

def install_alternative_numpy_setup():
    """Alternative setup dengan NumPy yang lebih kompatibel"""
    logger.info("=== ALTERNATIVE NUMPY SETUP ===")
    
    try:
        # Clear pip cache
        subprocess.check_call([sys.executable, '-m', 'pip', 'cache', 'purge'])
        
        # Install NumPy dengan binary wheel yang pre-compiled
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'numpy==1.24.3', '--only-binary=all', '--force-reinstall'
        ])
        
        # Install scientific stack yang kompatibel
        scientific_packages = [
            "scipy==1.10.1",
            "scikit-learn==1.3.0", 
            "pandas==2.0.3"
        ]
        
        for package in scientific_packages:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                package, '--only-binary=all'
            ])
        
        # Install OpenCV dengan versi yang kompatibel
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'opencv-python==4.8.0.76', '--only-binary=all'
        ])
        
        # Install TensorFlow versi yang kompatibel
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'tensorflow==2.13.1', '--only-binary=all'
        ])
        
        return True
        
    except Exception as e:
        logger.error(f"Alternative setup failed: {e}")
        return False

def test_imports():
    """Test import semua library penting"""
    logger.info("=== TESTING IMPORTS ===")
    
    test_imports = [
        ('numpy', 'import numpy as np; print(f"NumPy {np.__version__} OK")'),
        ('opencv', 'import cv2; print(f"OpenCV {cv2.__version__} OK")'),
        ('tensorflow', 'import tensorflow as tf; print(f"TensorFlow {tf.__version__} OK")'),
        ('sklearn', 'import sklearn; print(f"Scikit-learn {sklearn.__version__} OK")'),
        ('pandas', 'import pandas as pd; print(f"Pandas {pd.__version__} OK")'),
        ('PIL', 'import PIL; print(f"Pillow {PIL.__version__} OK")'),
        ('deepface', 'from deepface import DeepFace; print("DeepFace OK")'),
    ]
    
    results = {}
    
    for name, import_code in test_imports:
        try:
            result = subprocess.run([sys.executable, '-c', import_code], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                results[name] = "SUCCESS"
                logger.info(f"✅ {name}: {result.stdout.strip()}")
            else:
                results[name] = f"FAILED: {result.stderr.strip()}"
                logger.error(f"❌ {name}: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            results[name] = "TIMEOUT"
            logger.error(f"⏰ {name}: Import timeout")
        except Exception as e:
            results[name] = f"ERROR: {e}"
            logger.error(f"💥 {name}: {e}")
    
    return results

def test_deepface_functionality():
    """Test fungsionalitas DeepFace"""
    logger.info("=== TESTING DEEPFACE FUNCTIONALITY ===")
    
    test_code = '''
import numpy as np
from deepface import DeepFace
import cv2

# Test dengan dummy image
test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

try:
    # Test embedding generation
    embedding = DeepFace.represent(
        img_path=test_image,
        model_name="Facenet",
        detector_backend="opencv",
        enforce_detection=False
    )
    print(f"✅ FaceNet embedding: {len(embedding[0]['embedding'])} dimensions")
    
    # Test face detection
    faces = DeepFace.extract_faces(
        img_path=test_image,
        detector_backend="opencv",
        enforce_detection=False
    )
    print(f"✅ Face detection: {len(faces)} faces found")
    
    print("🎉 DeepFace functionality test PASSED")
    
except Exception as e:
    print(f"❌ DeepFace test FAILED: {e}")
'''
    
    try:
        result = subprocess.run([sys.executable, '-c', test_code], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.info("DeepFace functionality test PASSED")
            logger.info(result.stdout)
            return True
        else:
            logger.error("DeepFace functionality test FAILED")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Error testing DeepFace: {e}")
        return False

def create_fixed_requirements():
    """Create requirements.txt dengan versi yang sudah teruji"""
    requirements_content = '''# Fixed requirements untuk Python 3.9.23
# Mengatasi NumPy compatibility issues

# Core scientific computing (urutan penting)
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.0
pandas==2.0.3

# Image processing
Pillow==10.0.1
opencv-python==4.8.0.76

# Machine Learning
tensorflow==2.13.1

# Face Recognition
deepface==0.0.79
mtcnn==0.1.1
retina-face==0.0.13

# Django
Django==4.2.23
psycopg2-binary==2.9.9
channels==4.0.0
django-cors-headers==4.3.1

# Utilities
requests==2.31.0
tqdm==4.66.1
gdown==4.7.1
python-dotenv==1.0.0

# Development
pytest==7.4.3
pytest-django==4.7.0
colorlog==6.7.0
'''
    
    with open('requirements_fixed.txt', 'w') as f:
        f.write(requirements_content)
    
    logger.info("Created requirements_fixed.txt with compatible versions")

def main():
    """Main function untuk fix compatibility issues"""
    logger.info("=== NUMPY COMPATIBILITY FIX TOOL ===")
    logger.info(f"Python version: {sys.version}")
    
    # Check current versions
    current_versions = check_current_versions()
    
    # Create fixed requirements
    create_fixed_requirements()
    
    # Ask user confirmation
    print("\n" + "="*50)
    print("AKAN DILAKUKAN:")
    print("1. Uninstall semua packages yang bermasalah")
    print("2. Install ulang dengan versi yang kompatibel")
    print("3. Test functionality")
    print("="*50)
    
    response = input("Lanjutkan? (y/N): ").strip().lower()
    
    if response != 'y':
        logger.info("Operasi dibatalkan")
        return False
    
    # Uninstall problematic packages
    uninstall_problematic_packages()
    
    # Try main installation
    if install_compatible_versions():
        logger.info("Installation successful!")
    else:
        logger.warning("Main installation failed, trying alternative...")
        if not install_alternative_numpy_setup():
            logger.error("All installation methods failed!")
            return False
    
    # Test imports
    import_results = test_imports()
    
    # Test DeepFace functionality
    deepface_works = test_deepface_functionality()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("INSTALLATION SUMMARY")
    logger.info("="*50)
    
    success_count = sum(1 for result in import_results.values() if result == "SUCCESS")
    total_count = len(import_results)
    
    logger.info(f"Import tests: {success_count}/{total_count} passed")
    logger.info(f"DeepFace functionality: {'✅ PASSED' if deepface_works else '❌ FAILED'}")
    
    if success_count == total_count and deepface_works:
        logger.info("🎉 ALL TESTS PASSED! NumPy compatibility fixed!")
        logger.info("\nNext steps:")
        logger.info("1. python manage.py migrate")
        logger.info("2. python manage.py runserver")
        return True
    else:
        logger.error("❌ Some tests failed. Check logs above.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Restart your terminal/IDE")
        logger.info("2. Try: python -m pip install --upgrade pip")
        logger.info("3. Use virtual environment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)