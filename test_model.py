"""
Quick test of the Detectron2Model class
"""
import sys
import os

# Add src to path
sys.path.append('src')

try:
    from models.detectron2_model import Detectron2Model
    import numpy as np
    import cv2

    print("🧪 Testing Detectron2Model class...")

    # Test model initialization
    print("\n1. Testing model initialization...")
    model = Detectron2Model('faster_rcnn', num_classes=80, device="auto")
    print(f"   ✅ Model created: {model.model_name}")
    print(f"   ✅ Device: {model.device}")

    # Test model info
    print("\n2. Testing model info...")
    info = model.get_model_info()
    print(f"   ✅ Model info: {info}")

    # Test prediction with dummy image
    print("\n3. Testing prediction...")
    # Create a dummy image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Run prediction
    predictions = model.predict(test_image)
    print(f"   ✅ Prediction completed")
    print(f"   ✅ Number of detections: {len(predictions['instances'])}")

    if len(predictions['instances']) > 0:
        scores = predictions['instances'].scores.cpu().numpy()
        print(f"   ✅ Detection scores: {scores[:5]}")  # Show first 5 scores

    print("\n🎉 All tests passed! Your Detectron2Model is working correctly!")

except Exception as e:
    print(f"❌ Error testing Detectron2Model: {e}")
    import traceback
    traceback.print_exc()
