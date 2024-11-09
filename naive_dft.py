import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Naive_DFT_Implementation:
    def __init__(self):
        pass
    
    def naive_dft(self, signal) -> np.ndarray:
        N = len(signal)
        output = np.zeros(N, dtype=complex)
        
        for k in range(N):
            sum_val = 0j
            for n in range(N):
                angle = (-2 * np.pi * k * n) / N
                sum_val += signal[n] * (np.cos(angle) + 1j * np.sin(angle))
            output[k] = sum_val
                
        return output

    def cooley_tukey_fft(self, signal: np.ndarray, threshold: int = 8) -> np.ndarray:
        N = len(signal)
        
        # Base case: If N is small, use the naive DFT
        if N <= threshold:
            return self.naive_dft(signal)
        
        # Ensure N is a power of 2
        if N & (N - 1) != 0:
            raise ValueError("Input size must be a power of 2 for the Cooley-Tukey algorithm.")
        
        # Recursively apply FFT on even and odd indices
        even_part = self.cooley_tukey_fft(signal[::2], threshold)
        odd_part = self.cooley_tukey_fft(signal[1::2], threshold)
        
        # Precompute the twiddle factors
        factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        
        # Combine even and odd parts with twiddle factors
        combined = np.zeros(N, dtype=complex)
        for k in range(N // 2):
            combined[k] = even_part[k] + factor[k] * odd_part[k]
            combined[k + N // 2] = even_part[k] - factor[k] * odd_part[k]
        
        return combined

    def shift_zero_frequency(self, dft_result: np.ndarray) -> np.ndarray:
        M, N = dft_result.shape
        shifted = np.zeros_like(dft_result)
        
        for i in range(M):
            for j in range(N):
                new_i = (i + M//2) % M
                new_j = (j + N//2) % N
                shifted[new_i, new_j] = dft_result[i, j]
                
        return shifted

    def pad_to_power_of_2(self, image_array: np.ndarray) -> np.ndarray:
        # Find the next power of 2 for each dimension
        M, N = image_array.shape
        padded_M = 2**np.ceil(np.log2(M)).astype(int)
        padded_N = 2**np.ceil(np.log2(N)).astype(int)
        
        # Pad the image array to the new size
        padded_image = np.zeros((padded_M, padded_N), dtype=image_array.dtype)
        padded_image[:M, :N] = image_array
        return padded_image, M, N  # Return original size for trimming later

    def process_and_display_image(self, image_path: str):
        # Load and resize image if necessary
        image = Image.open(image_path).convert('L')
        
        # Resize large images for computational efficiency
        max_size = 256
        if max(image.size) > max_size:
            print(f"Resizing image to {max_size}x{max_size} for computational efficiency")
            image = image.resize((max_size, max_size))
        
        image_array = np.array(image)
        
        # Pad image to nearest power of 2
        padded_image, orig_M, orig_N = self.pad_to_power_of_2(image_array)
        
        # Compute the 2D FFT using the Cooley-Tukey algorithm
        dft_result = np.zeros_like(padded_image, dtype=complex)
        for i in range(padded_image.shape[0]):
            dft_result[i, :] = self.cooley_tukey_fft(padded_image[i].astype(float))
        for j in range(dft_result.shape[1]):
            dft_result[:, j] = self.cooley_tukey_fft(dft_result[:, j])

        dft_shifted = self.shift_zero_frequency(dft_result)
        
        # Calculate the magnitude spectrum (log scale) and trim to original size
        magnitude_spectrum = np.abs(dft_shifted)[:orig_M, :orig_N]
        
        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Display original image
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display unshifted Fourier transform
        unshifted_spectrum = np.abs(dft_result[:orig_M, :orig_N])
        spectrum_plot1 = axes[1].imshow(unshifted_spectrum, 
                                      norm=LogNorm(vmin=np.min(unshifted_spectrum[unshifted_spectrum > 0]), 
                                                 vmax=np.max(unshifted_spectrum)),
                                      cmap='gray')
        axes[1].set_title('Fourier Transform\n(Unshifted)')
        axes[1].axis('off')
        plt.colorbar(spectrum_plot1, ax=axes[1], label='Magnitude (log scale)')
        
        # Display shifted Fourier transform
        spectrum_plot2 = axes[2].imshow(magnitude_spectrum, 
                                      norm=LogNorm(vmin=np.min(magnitude_spectrum[magnitude_spectrum > 0]), 
                                                 vmax=np.max(magnitude_spectrum)),
                                      cmap='gray')
        axes[2].set_title('Fourier Transform\n(Zero-Frequency Shifted)')
        axes[2].axis('off')
        plt.colorbar(spectrum_plot2, ax=axes[2], label='Magnitude (log scale)')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dft = Naive_DFT_Implementation()
    dft.process_and_display_image('moonlanding.png')