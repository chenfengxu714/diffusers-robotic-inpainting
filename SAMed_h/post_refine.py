import cv2
import numpy as np
import os

def mask_refinement(image, mask, area_threshold=100):
    # Thresholding based on RGB ranges
    lower_bound = np.array([20, 20, 20])  # e.g., low thresholds for black
    upper_bound = np.array([250, 250, 250])  # e.g., high thresholds for white

    # Create a mask for colors within the specified range
    mask_color = cv2.inRange(image, lower_bound, upper_bound)

    # Apply morphological opening to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # # Remove small areas from the mask
    num_labels, labels_im = cv2.connectedComponents(mask)
    for i in range(1, num_labels):
        if np.sum(labels_im == i) < area_threshold:
            mask[labels_im == i] = 0

    return mask

def process_directory(image_dir, mask_dir):
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

    for img_path, mask_path in zip(images, masks):
        image = cv2.imread(img_path)
        original_mask = cv2.imread(mask_path, 0)  # Load mask in grayscale

        if image is not None and original_mask is not None:
            refined_mask = mask_refinement(image, original_mask)
            # Apply the mask to the image
            masked_image = cv2.bitwise_and(image, image, mask=refined_mask)

            # Save or display the result
            output_path = 'output_path/' + os.path.basename(img_path)
            cv2.imwrite(output_path, masked_image)
            # Optionally display the output
            # cv2.imshow('Masked Image', masked_image)
            # cv2.waitKey(0)


# Example usage
image_directory = '/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-04-15-drawer-adjusted-wrist_masks/Mon_Apr_15_02:03:19_2024/r2r_images'
mask_directory = '/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-04-15-drawer-adjusted-wrist_masks/Mon_Apr_15_02:03:19_2024/masks'
process_directory(image_directory, mask_directory)
