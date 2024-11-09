import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

"""
DO NOT SUBMIT THIS FILE!
I have written down the built-in FFT functions here to compare our result with the built-in functions.
"""

def process_image_fft(image_path: str):
    """
    Load an image, compute its FFT, and display the results.
    
    Args:
        image_path: Path to the input image file
    """
    # Load and convert image to grayscale
    image = Image.open(image_path).convert('L')
    
    max_size = 256
    if max(image.size) > max_size:
        print(f"Resizing image to {max_size}x{max_size} for computational efficiency")
        image = image.resize((max_size, max_size))
    
    image_array = np.array(image)
    
    # Compute 2D FFT
    fft_result = np.fft.fft2(image_array)
    
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Calculate magnitude spectrum (log scale)
    magnitude_spectrum = np.abs(fft_shifted)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    ax1.imshow(image_array, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Display FFT (log scale)
    spectrum_plot = ax2.imshow(magnitude_spectrum, 
                             norm=LogNorm(vmin=np.min(magnitude_spectrum[magnitude_spectrum > 0]), 
                                        vmax=np.max(magnitude_spectrum)),
                             cmap='gray')
    ax2.set_title('Fourier Transform (Log Scale)')
    ax2.axis('off')
    
    # Add colorbar
    plt.colorbar(spectrum_plot, ax=ax2, label='Magnitude (log scale)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_image_fft('moonlanding.png')