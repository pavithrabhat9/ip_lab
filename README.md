import cv2
import numpy as np
import os

# Define the user-provided image path
image_path = r"D:\images.jpeg"

# Load the original image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    print(f"Error: Could not read image from {image_path}")
    exit()

# Apply Gaussian Blur to create a blurry image
kernel_size = (21, 21)  # Adjust size as needed
sigma = 5  # Standard deviation
gaussian_blur = cv2.GaussianBlur(img_bgr, kernel_size, sigma)
cv2.imwrite("D:\blurry_image.png", gaussian_blur)

# Generate a Gaussian blur kernel
kernel_1d = cv2.getGaussianKernel(ksize=kernel_size[0], sigma=sigma)
gaussian_kernel = kernel_1d * kernel_1d.T  # Convert to 2D
cv2.imwrite("D:\blur_kernel.png", gaussian_kernel * 255)  # Save kernel as an image

# Load the blur kernel
h = cv2.imread("D:\blur_kernel.png", 0)
if h is None:
    print("Error: Could not read blur kernel")
    exit()

# Resize kernel to match the input image dimensions
h_resized = cv2.resize(h, (img_bgr.shape[1], img_bgr.shape[0]))

# Initialize restored image
restored = np.zeros(img_bgr.shape)

for i in range(3):  # Loop through color channels
    g = img_bgr[:, :, i]
    G = np.fft.fft2(g)
    h_padded = np.zeros(g.shape)
    h_padded[:h_resized.shape[0], :h_resized.shape[1]] = np.copy(h_resized)
    H = np.fft.fft2(h_padded)
    
    H_norm = H / abs(H.max())
    G_norm = G / abs(G.max())
    F_temp = G_norm / H_norm
    F_norm = F_temp / abs(F_temp.max())

    F_hat = F_norm * abs(G.max())
    f_hat = np.fft.ifft2(F_hat)
    restored[:, :, i] = abs(f_hat)

# Save the restored output
output_path = "D:\restored_image.png"
cv2.imwrite(output_path, restored)
print(f"Restored image saved at {output_path}")

I've modified the code to apply Gaussian blur directly to the user's provided image instead of generating a new one. It now:

1. Reads the given image (D:\images.jpeg).


2. Applies Gaussian blur to create a blurry version (D:\blurry_image.png).


3. Generates a Gaussian blur kernel (D:\blur_kernel.png).


4. Performs inverse filtering to restore the image.



Let me know if you need any further refinements! 🚀

