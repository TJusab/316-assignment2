import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

def process_image_fft(image_path: str):
    """
    Load an image, compute its FFT, and display the results.
    
    Args:
        image_path: Path to the input image file
    """
    # Load and convert image to grayscale
    image = Image.open(image_path).convert('L')
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
    try:
        process_image_fft('moonlanding.png')
    except FileNotFoundError:
        print("Image not found. Creating and processing a test pattern...")
        
        # Create a simple test pattern
        test_size = 256
        test_pattern = np.zeros((test_size, test_size))
        test_pattern[test_size//4:3*test_size//4, test_size//4:3*test_size//4] = 1
        
        plt.imsave('test_pattern.png', test_pattern, cmap='gray')
        print("Created test pattern 'test_pattern.png'")
        process_image_fft('test_pattern.png')