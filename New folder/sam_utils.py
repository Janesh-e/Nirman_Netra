import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Initialize SAM model
def initialize_sam(model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth", device="cuda"):
    """
    Initialize the SAM model.
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

# Generate masks using bounding boxes
def generate_masks_with_boxes(predictor, image_path, bounding_boxes):
    """
    Generate masks using bounding boxes as input prompts.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set the image in the predictor
    predictor.set_image(image)

    # Convert bounding boxes to the required format
    bounding_boxes = np.array(bounding_boxes, dtype=np.float32)  # Convert list to NumPy array

    print(f"Original Bounding Boxes Shape: {bounding_boxes.shape}")

    # Convert to PyTorch tensor and move to the same device as the model
    #bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32, device=predictor.device)
    bounding_boxes = torch.from_numpy(bounding_boxes).to(predictor.device)  # Convert to PyT
    
    # Ensure bounding boxes are (N, 4)
    #if bounding_boxes.ndim == 1:
        #bounding_boxes = bounding_boxes.unsqueeze(0)  # Add batch dimension

    # Check shape before passing to the predictor
    print("Processed Bounding Boxes Shape:", bounding_boxes.shape)  # Debugging statement
    print("Processed Bounding Boxes" , bounding_boxes)
    # List to store individual masks
    all_masks = []

    # Process each bounding box separately
    for box in bounding_boxes:
        single_box = box.unsqueeze(0)  # Reshape to (1, 4)

        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=single_box,
            multimask_output=False,
        )

        # Convert to tensor if needed
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32, device=predictor.device)

        all_masks.append(mask)

    # Combine all masks into a single final mask
    if len(all_masks) > 1:
        final_mask = torch.stack(all_masks).sum(dim=0)  # Sum all masks
        final_mask = final_mask.clamp(0, 1)  # Ensure values remain between 0 and 1
    else:
        final_mask = all_masks[0]  # If only one mask, use it directly

    return final_mask




    # Ensure bounding boxes are in the correct shape (N, 4)
    if bounding_boxes.ndim == 1:
        bounding_boxes = bounding_boxes.reshape(-1, 4)
    # Convert bounding boxes to the required format
    #input_boxes = torch.tensor(bounding_boxes, device=predictor.device)

    # Generate masks
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bounding_boxes,
        multimask_output=False,
    )

    return masks



# Generate masks using single points
def generate_masks_with_points(predictor, image_path, points):
    """
    Generate masks using single points as input prompts.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set the image in the predictor
    predictor.set_image(image)

    # Convert points to the required format
    input_points = torch.tensor(points, device=predictor.device)
    input_labels = torch.ones(input_points.shape[0], device=predictor.device)  # All points are foreground

    # Generate masks
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=None,
        multimask_output=False,
    )

    return masks

# Save mask as an image
def save_mask_as_image(mask, output_path):
    """
    Save a mask as an image.
    """
    if isinstance(mask, torch.Tensor):  # Convert PyTorch tensor to NumPy array
        mask = mask.cpu().numpy()
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_image)