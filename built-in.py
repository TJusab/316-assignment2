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
    
    image_array = np.array(image)
    
    M, N = image_array.shape
    padded_M = 2**np.ceil(np.log2(M)).astype(int)
    padded_N = 2**np.ceil(np.log2(N)).astype(int)
    
    # Create a zero array of the target size
    padded_image = np.zeros((padded_M, padded_N), dtype=image_array.dtype)
    
    # Place the original image in the center
    padded_image[:M, :N] = image_array
    
    # Compute 2D FFT
    fft_result = np.fft.fft2(padded_image)
    
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
    
def denoise(image_array: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
    # Perform 2D FFT on the image using numpy's fft2
    fft_result = np.fft.fftshift(np.fft.fft2(image_array))
    
    # Calculate cutoff frequencies
    M, N = fft_result.shape
    cutoff_x = int(M * cutoff_ratio / 2)
    cutoff_y = int(N * cutoff_ratio / 2)
    
    # Create a mask for the low frequencies (central square) and apply it
    mask = np.zeros_like(fft_result)
    mask[M//2 - cutoff_x : M//2 + cutoff_x, N//2 - cutoff_y : N//2 + cutoff_y] = 1
    filtered_fft = fft_result * mask
    
    # Count non-zero coefficients in the filtered FFT and calculate fraction retained
    non_zero_count = np.count_nonzero(filtered_fft)
    total_coefficients = M * N
    fraction_retained = non_zero_count / total_coefficients
    
    # Print the non-zero coefficient count and fraction retained
    print(f"Non-zero coefficients retained: {non_zero_count}")
    print(f"Fraction of original Fourier coefficients retained: {fraction_retained:.4f}")
    
    # Perform the inverse FFT to obtain the denoised image
    denoised_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
    
    # Return the real part of the denoised image
    return denoised_image

if __name__ == "__main__":
    process_image_fft("test.jpg")
    
    # denoised_image = denoise(image_array, cutoff_ratio=0.2)
    # plt.imshow(denoised_image, cmap='gray')
    # plt.title('Denoised Image')
    # plt.axis('off')
    # plt.show()