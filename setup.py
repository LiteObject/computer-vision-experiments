from setuptools import setup, find_packages

setup(
    name='cv-yolo-experiments',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for experimenting with YOLO models for computer vision tasks.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy',
        'opencv-python',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'Pillow',
        'jupyter',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)