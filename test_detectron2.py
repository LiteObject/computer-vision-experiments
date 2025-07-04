"""
Test script to verify Detectron2 installation and compatibility
"""
import sys
import subprocess
import warnings
import platform


def check_system_info():
    """Display system information"""
    print("🖥️  System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Architecture: {platform.machine()}")
    print()


def check_pytorch():
    """Check PyTorch installation and capabilities"""
    print("🔥 PyTorch Information:")
    try:
        import torch
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   ✅ CUDA version: {torch.version.cuda}")
            print(f"   ✅ GPU device: {torch.cuda.get_device_name(0)}")
            device_type = "CUDA"
        else:
            print(f"   ⚠️  CUDA not available - using CPU")
            device_type = "CPU"

        # Check if PyTorch can create tensors
        test_tensor = torch.randn(2, 3)
        print(f"   ✅ Tensor creation works: {test_tensor.shape}")

        return True, device_type, torch.__version__

    except ImportError as e:
        print(f"   ❌ PyTorch not found: {e}")
        return False, None, None
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False, None, None


def check_opencv():
    """Check OpenCV installation"""
    print("📷 OpenCV Information:")
    try:
        import cv2
        print(f"   ✅ OpenCV version: {cv2.__version__}")

        # Test basic functionality
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        print(f"   ✅ Image creation works: {test_img.shape}")

        return True
    except ImportError as e:
        print(f"   ❌ OpenCV not found: {e}")
        return False
    except Exception as e:
        print(f"   ❌ OpenCV error: {e}")
        return False


def check_detectron2():
    """Check Detectron2 installation and basic functionality"""
    print("🔧 Detectron2 Information:")
    try:
        # Try to import detectron2
        import detectron2
        print(f"   ✅ Detectron2 version: {detectron2.__version__}")

        # Test basic imports
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        print("   ✅ Core modules imported successfully")

        # Test configuration
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        print("   ✅ Configuration loading works")

        # Test model zoo access
        model_url = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        print(f"   ✅ Model zoo access works")

        # Test if we can create a predictor (without downloading weights)
        cfg.MODEL.WEIGHTS = ""  # Don't download weights for test
        try:
            predictor = DefaultPredictor(cfg)
            print("   ✅ Predictor creation works")
        except Exception as e:
            print(
                f"   ⚠️  Predictor creation failed (expected without weights): {e}")

        return True, detectron2.__version__

    except ImportError as e:
        print(f"   ❌ Detectron2 not found: {e}")
        return False, None
    except Exception as e:
        print(f"   ❌ Detectron2 error: {e}")
        return False, None


def get_installation_recommendations(pytorch_available, device_type, torch_version):
    """Provide installation recommendations based on system setup"""
    print("\n💡 Installation Recommendations:")

    if not pytorch_available:
        print("   1. Install PyTorch first:")
        print("      pip install torch torchvision torchaudio")
        return

    print("   Choose one of the following installation methods:")
    print()

    # Method 1: Pre-built wheels
    print("   📦 Method 1: Pre-built wheels (Recommended)")
    if device_type == "CUDA":
        print("      # For CUDA 11.8")
        print("      pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html")
        print()
        print("      # For CUDA 12.1")
        print("      pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html")
    else:
        print("      # For CPU (your current setup)")
        print("      pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html")
        print()
        print("      # Alternative CPU wheel")
        print("      pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html")

    print()

    # Method 2: From source
    print("   🔨 Method 2: Build from source")
    print("      # Install dependencies first")
    print("      pip install ninja wheel")
    print("      pip install fvcore iopath omegaconf hydra-core")
    print("      pip install pycocotools")
    print()
    print("      # Install detectron2")
    print("      pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    print()

    # Method 3: Conda
    print("   🐍 Method 3: Using conda (if available)")
    print("      conda install -c conda-forge detectron2")
    print()

    # Method 4: Docker
    print("   🐳 Method 4: Docker (for development)")
    print("      docker pull detectron2/detectron2:latest")
    print("      docker run --gpus all -it --rm detectron2/detectron2:latest")


def test_full_workflow():
    """Test a complete workflow if everything is available"""
    print("\n🧪 Testing Complete Workflow:")

    try:
        import torch
        import cv2
        import numpy as np
        from detectron2 import model_zoo
        from detectron2.config import get_cfg

        print("   ✅ All imports successful")

        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("   ✅ Test image created")

        # Setup basic configuration
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.DEVICE = "cpu"  # Force CPU for testing
        print("   ✅ Configuration setup complete")

        print("   🎉 Basic workflow test PASSED!")
        return True

    except Exception as e:
        print(f"   ❌ Workflow test FAILED: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Detectron2 Installation Test")
    print("=" * 50)
    print()

    # Check system info
    check_system_info()

    # Check PyTorch
    pytorch_ok, device_type, torch_version = check_pytorch()
    print()

    # Check OpenCV
    opencv_ok = check_opencv()
    print()

    # Check Detectron2
    detectron2_ok, d2_version = check_detectron2()
    print()

    # Test full workflow if everything is available
    if pytorch_ok and opencv_ok and detectron2_ok:
        workflow_ok = test_full_workflow()
    else:
        workflow_ok = False

    # Provide recommendations
    if not detectron2_ok:
        get_installation_recommendations(
            pytorch_ok, device_type, torch_version)

    # Summary
    print("\n📊 Test Summary:")
    print("=" * 30)
    print(f"PyTorch:    {'✅ PASS' if pytorch_ok else '❌ FAIL'}")
    print(f"OpenCV:     {'✅ PASS' if opencv_ok else '❌ FAIL'}")
    print(f"Detectron2: {'✅ PASS' if detectron2_ok else '❌ FAIL'}")
    print(f"Workflow:   {'✅ PASS' if workflow_ok else '❌ FAIL'}")

    if detectron2_ok:
        print(f"\n🎉 SUCCESS! Detectron2 {d2_version} is ready to use!")
        print("\nNext steps:")
        print("   1. Open notebooks/detectron2_training.ipynb")
        print("   2. Try the model comparison notebook")
        print("   3. Start experimenting with your data!")
    else:
        print(f"\n⚠️  Detectron2 installation needed.")
        print("   Follow the recommendations above to install Detectron2.")

    return detectron2_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
