import json
import numpy as np
import cv2
from typing import List, Dict, Tuple
import os
import requests


def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")


def save_results(results, filename):
    """Saves results to a specified file."""
    with open(filename, 'w') as f:
        f.write(str(results))
    log_message(f"Results saved to {filename}")


def load_results(filename):
    """Loads results from a specified file."""
    with open(filename, 'r') as f:
        results = f.read()
    log_message(f"Results loaded from {filename}")
    return results


def create_directory(directory):
    """Creates a directory if it does not exist."""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
        log_message(f"Directory created: {directory}")
    else:
        log_message(f"Directory already exists: {directory}")


def download_sample_data(output_dir="./sample_data"):
    """Download sample images for testing"""
    create_directory(output_dir)

    # Sample image URLs (replace with actual URLs or use local images)
    sample_urls = [
        "https://via.placeholder.com/640x480.jpg?text=Sample+Image+1",
        "https://via.placeholder.com/640x480.jpg?text=Sample+Image+2"
    ]

    downloaded_files = []
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                filename = os.path.join(output_dir, f"sample_{i+1}.jpg")
                with open(filename, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(filename)
                log_message(f"Downloaded: {filename}")
        except Exception as e:
            log_message(f"Failed to download {url}: {e}")

    return downloaded_files


def prepare_coco_dataset(images_dir: str, annotations_file: str, output_dir: str = "./coco_dataset"):
    """Prepare a COCO format dataset for Detectron2"""
    create_directory(output_dir)

    # Copy images to output directory
    import shutil
    output_images_dir = os.path.join(output_dir, "images")
    create_directory(output_images_dir)

    try:
        if os.path.exists(images_dir):
            shutil.copytree(images_dir, output_images_dir, dirs_exist_ok=True)
            log_message(f"Images copied to {output_images_dir}")
    except Exception as e:
        log_message(f"Error copying images: {e}")

    # Copy annotations file
    if os.path.exists(annotations_file):
        shutil.copy2(annotations_file, os.path.join(
            output_dir, "annotations.json"))
        log_message(f"Annotations copied to {output_dir}")

    return output_dir


def convert_yolo_to_coco(yolo_annotations_dir: str, images_dir: str, output_file: str, class_names: List[str]):
    """Convert YOLO format annotations to COCO format"""
    import glob

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for i, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": i + 1,
            "name": class_name,
            "supercategory": "object"
        })

    annotation_id = 1

    # Process each image
    for img_path in glob.glob(os.path.join(images_dir, "*.jpg")):
        img_name = os.path.basename(img_path)
        img_id = len(coco_data["images"]) + 1

        # Read image to get dimensions
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        # Look for corresponding YOLO annotation file
        yolo_file = os.path.join(yolo_annotations_dir,
                                 img_name.replace(".jpg", ".txt"))
        if os.path.exists(yolo_file):
            with open(yolo_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, w, h = map(float, parts)

                        # Convert YOLO to COCO format
                        x = (x_center - w/2) * width
                        y = (y_center - h/2) * height
                        bbox_width = w * width
                        bbox_height = h * height

                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": int(class_id) + 1,
                            "bbox": [x, y, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0
                        })
                        annotation_id += 1

    # Save COCO format annotations
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    log_message(f"COCO annotations saved to {output_file}")
    return output_file


def compare_model_outputs(predictions_dict: Dict, image_path: str, save_path: str = None):
    """Compare outputs from different models on the same image"""
    import matplotlib.pyplot as plt

    num_models = len(predictions_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))

    if num_models == 1:
        axes = [axes]

    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        # Visualize predictions (this is a simplified version)
        vis_image = image_rgb.copy()

        # Add bounding boxes (format depends on model type)
        if 'instances' in predictions:  # Detectron2 format
            instances = predictions['instances']
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()

            for box, score in zip(boxes, scores):
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(vis_image, (x1, y1),
                                  (x2, y2), (255, 0, 0), 2)
                    cv2.putText(vis_image, f'{score:.2f}', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        axes[i].imshow(vis_image)
        axes[i].set_title(
            f'{model_name}\n{len(predictions.get("instances", []))} detections')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log_message(f"Comparison saved to {save_path}")

    plt.show()


def create_model_summary_table(models_info: List[Dict]) -> str:
    """Create a formatted summary table of model information"""
    table = "| Model | Type | Classes | Device | FPS | Architecture |\n"
    table += "|-------|------|---------|--------|-----|-------------|\n"

    for info in models_info:
        table += f"| {info.get('model_name', 'Unknown')} | "
        table += f"{info.get('type', 'Unknown')} | "
        table += f"{info.get('num_classes', 'N/A')} | "
        table += f"{info.get('device', 'N/A')} | "
        table += f"{info.get('fps', 'N/A'):.1f} | " if isinstance(
            info.get('fps'), (int, float)) else "N/A | "
        table += f"{info.get('architecture', 'Unknown')} |\n"

    return table
