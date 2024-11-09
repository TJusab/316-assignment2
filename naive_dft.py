import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Naive_DFT_Implementation:
    def __init__(self):
        pass
        
    def naive_dft_2d(self, image_array: np.ndarray) -> np.ndarray:
        M, N = image_array.shape
        output = np.zeros((M, N), dtype=complex)
        
        # Apply DFT to rows
        transformed = np.zeros((M, N), dtype=complex)
        for i in range(M):
            transformed[i] = self.naive_dft(image_array[i].astype(float))
            
        # Apply DFT to cols
        for j in range(N):
            output[:, j] = self.naive_dft(transformed[:, j])
            
        return output
    
    def naive_dft(self, signal) -> np.ndarray:
        N = len(signal)
        output = np.zeros(N, dtype=complex)
        
        # For each frequency K
        for k in range(N):
            sum_val = 0j
            
            # Compute the sum from 0 to N - 1 
            for n in range(N):
                angle = (-2 * np.pi * k * n) / N
                sum_val += signal[n] * (np.cos(angle) + 1j * np.sin(angle))
            
            output[k] = sum_val
                
        return output
    
    def shift_zero_frequency(self, dft_result: np.ndarray) -> np.ndarray:
        """
        Manually shift zero frequency component to center.
        TODO Check with prof if this is necessary
        """
        M, N = dft_result.shape
        shifted = np.zeros_like(dft_result)
        
        # For each point in the original DFT
        for i in range(M):
            for j in range(N):
                # Calculate new indices with shift
                new_i = (i + M//2) % M
                new_j = (j + N//2) % N
                # Move the point to its new position
                shifted[new_i, new_j] = dft_result[i, j]
                
        return shifted
    
    def process_and_display_image(self, image_path: str):
        # Load and resize image if necessary
        image = Image.open(image_path).convert('L')
        
        # Resize large images for computational efficiency
        max_size = 64
        if max(image.size) > max_size:
            print(f"Resizing image to {max_size}x{max_size} for computational efficiency")
            image = image.resize((max_size, max_size))
            
        image_array = np.array(image)
        
        # Compute the 2D DFT image
        dft_result = self.naive_dft_2d(image_array)
        
        # Manually shift zero frequency to center
        dft_shifted = self.shift_zero_frequency(dft_result)
        
        # Calculate the magnitude spectrum (log scale)
        magnitude_spectrum = np.abs(dft_shifted)
        
        # Create visualizations to show the difference
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Display original image
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display unshifted Fourier transform
        unshifted_spectrum = np.abs(dft_result)
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