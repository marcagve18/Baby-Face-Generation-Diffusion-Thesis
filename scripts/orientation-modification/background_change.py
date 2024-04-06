import cv2
import numpy as np

def change_background(foreground_path, background_path, output_path):
    # Load the foreground image
    foreground = cv2.imread(foreground_path)
    
    # Convert the foreground image to grayscale
    gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to create a mask (binary image)
    # Adjust the threshold value as needed for your image
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphology operations to refine the mask
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 1)
    
    # Load the background image (ensure it is the same size as the foreground image)
    background = cv2.imread(background_path)
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    
    # Invert mask to get the background mask
    background_mask = cv2.bitwise_not(mask)
    
    # Use the masks to extract the relevant parts from fg and bg images
    foreground_part = cv2.bitwise_and(foreground, foreground, mask=mask)
    background_part = cv2.bitwise_and(background, background, mask=background_mask)
    
    # Combine the foreground and background parts
    combined = cv2.add(foreground_part, background_part)
    
    # Save the output image
    cv2.imwrite(output_path, combined)

# Example usage
foreground_path = '/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/orientation/img_2024-03-09 13:27:41.362387_Lateral portrait of a asian smiling baby, lateral view, profile, side.png'
background_path = '/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/tests/bg1.webp'
output_path = 'test.jpg'

change_background(foreground_path, background_path, output_path)
