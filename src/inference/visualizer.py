def draw_bounding_boxes(image, boxes, scores, classes, class_names, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes on the image.

    Parameters:
    - image: The image on which to draw.
    - boxes: List of bounding boxes (x1, y1, x2, y2).
    - scores: List of confidence scores for each box.
    - classes: List of class indices for each box.
    - class_names: List of class names corresponding to class indices.
    - color: Color of the bounding box.
    - thickness: Thickness of the bounding box lines.
    """
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        label = f"{class_names[cls]}: {score:.2f}"
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        image = cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def visualize_inference(image, boxes, scores, classes, class_names):
    """
    Visualizes the inference results by drawing bounding boxes on the image.

    Parameters:
    - image: The input image.
    - boxes: List of bounding boxes.
    - scores: List of confidence scores.
    - classes: List of class indices.
    - class_names: List of class names.
    """
    image_with_boxes = draw_bounding_boxes(image, boxes, scores, classes, class_names)
    cv2.imshow("Inference Result", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()