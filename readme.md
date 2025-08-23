# 1. NumPy DULU - sangat penting!
pip install numpy==1.24.3 --no-cache-dir --force-reinstall

# 2. Scientific stack
pip install scipy==1.10.1 --no-cache-dir
pip install scikit-learn==1.3.0 --no-cache-dir
pip install pandas==2.0.3 --no-cache-dir
pip install matplotlib==3.7.2 --no-cache-dir

# 3. Image processing
pip install Pillow==10.0.1 --no-cache-dir
pip install opencv-python==4.8.0.76 --no-cache-dir

# 4. TensorFlow dengan binary wheel
pip install tensorflow==2.12 --only-binary=all --no-cache-dir

pip install deepface==0.0.95 --no-cache-dir

pip install Django==4.2.23 --no-cache-dir
pip install psycopg2-binary==2.9.9 --no-cache-dir
pip install channels==4.0.0 --no-cache-dir
pip install django-cors-headers==4.3.1 --no-cache-dir
pip install requests==2.31.0 --no-cache-dir

# 5. Test TensorFlow
python -c "import tensorflow as tf; from deepface import DeepFace; print(f'TF: {tf.__version__}, DeepFace: OK')"

