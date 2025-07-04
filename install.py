#!/usr/bin/env python3
"""
Installation script for Computer Vision Experiments project.
This script helps install dependencies including Detectron2.
"""

import subprocess
import sys
import os
import platform


def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(command, shell=True, check=True,
                                capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - will use CPU")
        return cuda_available
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False


def install_pytorch():
    """Install PyTorch with CUDA support if available"""
    print("\n🔧 Installing PyTorch...")

    system = platform.system().lower()

    if system == "windows":
        # Windows with CUDA 11.8
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    elif system == "linux":
        # Linux with CUDA 11.8
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        # macOS or CPU-only
        command = "pip install torch torchvision torchaudio"

    return run_command(command, "Installing PyTorch")


def install_basic_requirements():
    """Install basic requirements"""
    print("\n🔧 Installing basic requirements...")
    return run_command("pip install -r requirements.txt", "Installing basic requirements")


def install_detectron2():
    """Install Detectron2"""
    print("\n🔧 Installing Detectron2...")

    # First install dependencies
    deps_command = "pip install fvcore iopath omegaconf hydra-core"
    if not run_command(deps_command, "Installing Detectron2 dependencies"):
        return False

    # Install pycocotools
    coco_command = "pip install pycocotools"
    if not run_command(coco_command, "Installing pycocotools"):
        return False

    # Install Detectron2
    system = platform.system().lower()

    if system == "windows":
        # For Windows, use pre-built wheels
        d2_command = "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html"
    else:
        # For Linux/macOS, build from source
        d2_command = "pip install 'git+https://github.com/facebookresearch/detectron2.git'"

    return run_command(d2_command, "Installing Detectron2")


def install_yolo_dependencies():
    """Install YOLO dependencies"""
    print("\n🔧 Installing YOLO dependencies...")

    commands = [
        ("pip install ultralytics", "Installing Ultralytics (YOLOv8)"),
        ("pip install yolov5", "Installing YOLOv5")
    ]

    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  Failed to install {description}")


def verify_installation():
    """Verify that key packages are installed correctly"""
    print("\n🔍 Verifying installation...")

    packages_to_check = [
        ("torch", "PyTorch"),
        ("detectron2", "Detectron2"),
        ("ultralytics", "Ultralytics (YOLOv8)"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy")
    ]

    success_count = 0
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {name} installed successfully")
            success_count += 1
        except ImportError:
            print(f"❌ {name} not found")

    print(
        f"\n📊 Installation Summary: {success_count}/{len(packages_to_check)} packages installed")
    return success_count == len(packages_to_check)


def main():
    """Main installation function"""
    print("🚀 Computer Vision Experiments - Installation Script")
    print("This script will install all dependencies including Detectron2, YOLO, and PyTorch")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install PyTorch first
    if not install_pytorch():
        print("❌ Failed to install PyTorch. Please install manually.")
        sys.exit(1)

    # Check CUDA after PyTorch installation
    check_cuda()

    # Install basic requirements
    if not install_basic_requirements():
        print("❌ Failed to install basic requirements.")
        sys.exit(1)

    # Install Detectron2
    if not install_detectron2():
        print("⚠️  Detectron2 installation failed. You can try installing manually:")
        print("   https://detectron2.readthedocs.io/en/latest/tutorials/install.html")

    # Install YOLO dependencies
    install_yolo_dependencies()

    # Verify installation
    if verify_installation():
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Open notebooks/detectron2_training.ipynb to get started with Detectron2")
        print("2. Open notebooks/model_comparison.ipynb to compare different models")
        print("3. Check out the README.md for more information")
    else:
        print("\n⚠️  Installation completed with some issues.")
        print(
            "Please check the error messages above and install missing packages manually.")


if __name__ == "__main__":
    main()
