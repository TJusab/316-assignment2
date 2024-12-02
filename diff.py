import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fft_impl import FFT_Implementation  # Assuming your previous implementation is in this file

def compare_fft_implementations(image_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
        
    image_array = np.array(image)
    
    M, N = image_array.shape
    padded_M = 2**np.ceil(np.log2(M)).astype(int)
    padded_N = 2**np.ceil(np.log2(N)).astype(int)
    
    # Create a zero array of the target size
    padded_image = np.zeros((padded_M, padded_N), dtype=image_array.dtype)
    
    # Place the original image in the center
    padded_image[:M, :N] = image_array
    
    # Create an instance of your custom FFT implementation
    fft_impl = FFT_Implementation()
    
    # Compute FFT using custom implementation
    custom_fft_2d = fft_impl.fft_2d(padded_image)
    custom_shifted_fft = fft_impl.shift_zero_frequency(custom_fft_2d)
    
    # Compute FFT using NumPy
    numpy_fft_2d = np.fft.fft2(padded_image)
    numpy_shifted_fft = np.fft.fftshift(numpy_fft_2d)
    
    # Compute absolute differences
    magnitude_diff = np.abs(np.abs(custom_shifted_fft) - np.abs(numpy_shifted_fft))
    phase_diff = np.abs(np.angle(custom_shifted_fft) - np.angle(numpy_shifted_fft))
    
    # Compute statistical metrics
    print("Magnitude Comparison:")
    print(f"Mean absolute difference: {np.mean(magnitude_diff)}")
    print(f"Max absolute difference: {np.max(magnitude_diff)}")
    print(f"Root Mean Square difference: {np.sqrt(np.mean(magnitude_diff**2))}")
    print(f"Relative difference: {np.mean(magnitude_diff) / np.mean(np.abs(numpy_shifted_fft)) * 100}%")
    
    print("\nPhase Comparison:")
    print(f"Mean phase difference: {np.mean(phase_diff)}")
    print(f"Max phase difference: {np.max(phase_diff)}")
    
    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original Image
    axs[0, 0].imshow(image_array, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    # Magnitude Spectrums (log scale)
    from matplotlib.colors import LogNorm
    
    custom_magnitude = np.abs(custom_shifted_fft)
    numpy_magnitude = np.abs(numpy_shifted_fft)
    
    im1 = axs[0, 1].imshow(np.log1p(custom_magnitude), 
                            norm=LogNorm(), cmap='viridis')
    axs[0, 1].set_title('Custom FFT Magnitude (Log)')
    axs[0, 1].axis('off')
    plt.colorbar(im1, ax=axs[0, 1])
    
    im2 = axs[0, 2].imshow(np.log1p(numpy_magnitude), 
                            norm=LogNorm(), cmap='viridis')
    axs[0, 2].set_title('NumPy FFT Magnitude (Log)')
    axs[0, 2].axis('off')
    plt.colorbar(im2, ax=axs[0, 2])
    
    # Difference Visualizations
    im3 = axs[1, 0].imshow(magnitude_diff, cmap='hot')
    axs[1, 0].set_title('Magnitude Absolute Difference')
    axs[1, 0].axis('off')
    plt.colorbar(im3, ax=axs[1, 0])
    
    im4 = axs[1, 1].imshow(phase_diff, cmap='cool')
    axs[1, 1].set_title('Phase Absolute Difference')
    axs[1, 1].axis('off')
    plt.colorbar(im4, ax=axs[1, 1])
    
    # Histogram of Differences
    axs[1, 2].hist(magnitude_diff.flatten(), bins=50, color='blue', alpha=0.7)
    axs[1, 2].set_title('Magnitude Difference Distribution')
    axs[1, 2].set_xlabel('Absolute Difference')
    axs[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Numerical Comparisons
    print("\nNumerical Comparison:")
    print(f"Are results equal within tolerance? {np.allclose(custom_shifted_fft, numpy_shifted_fft, rtol=1e-5, atol=1e-8)}")

if __name__ == "__main__":
    compare_fft_implementations("moonlanding.png")