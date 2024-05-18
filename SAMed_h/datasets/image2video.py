import cv2
import os

def images_to_video(image_folder, output_video, fps=2):
    # Get all image files from the input folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort the images by name

    # Assuming all images are the same size, get dimensions from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()  # Release the video writer

# Usage
image_folder = '/rscratch/cfxu/diffusion-RL/style-transfer/SAMed/SAMed_h/samh_output_checktimeconsistency/r2r'  # Update this path
output_video = 'output_video.mp4'  # Specify the output video file name
images_to_video(image_folder, output_video)
