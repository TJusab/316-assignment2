import argparse
from fft_impl import FFT_Implementation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
class FFT_Program:
    def __init__ (self, mode, image_path, fft):
        self.mode = mode
        self.image_path = image_path
        self.fft = fft
        self.image_array = None
        self.orig_M = None
        self.orig_N = None
        self.process_image()
        
    def process_image(self) :
        # Load image
        image = Image.open(self.image_path).convert('L')
        
        self.image_array = np.array(image)
        padded_image, self.orig_M, self.orig_N = self.pad_to_power_of_2()
    
        if self.mode == "1":
            fft_result = self.fft.shift_zero_frequency(self.fft.fft_2d(padded_image))
            self.plot_FFT(fft_result)
        elif self.mode == "2":
            fft_denoised = self.fft.denoise(padded_image)
            self.plot_denoised(fft_denoised)
        elif self.mode == "3":
            ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.99]
            images, counts = self.fft.compress_threshold(padded_image, ratios)
            self.plot_compressed_images(images, counts, ratios)
        elif self.mode == "4":
            self.fft.benchmark_performance()
            
        
    def pad_to_power_of_2(self):
        M, N = self.image_array.shape
        padded_M = 2**np.ceil(np.log2(M)).astype(int)
        padded_N = 2**np.ceil(np.log2(N)).astype(int)
    
        # Create a zero array of the target size
        padded_image = np.zeros((padded_M, padded_N), dtype=self.image_array.dtype)
    
        # Place the original image in the center
        padded_image[:M, :N] = self.image_array
    
        return padded_image, M, N

    def plot_FFT(self, fft_result):
        # Calculate magnitude spectrum (log scale)
        magnitude_spectrum = np.abs(fft_result)[:self.orig_M, :self.orig_N]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        # Display original image
        axes[0].imshow(self.image_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display Fourier Transform
        spectrum_plot = axes[1].imshow(magnitude_spectrum,
                                       norm=LogNorm(vmin=np.min(magnitude_spectrum[magnitude_spectrum > 0]),
                                                    vmax=np.max(magnitude_spectrum)),
                                       cmap='gray')
        
        axes[1].set_title('Fourier Transform')
        axes[1].axis('off')
        plt.colorbar(spectrum_plot, ax=axes[1], label='Magnitude (log scale)')
        
        plt.tight_layout()
        plt.show()
        
    def plot_denoised(self, fft_denoised):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        # Display original image
        axes[0].imshow(self.image_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
        axes[1].imshow(fft_denoised[:self.orig_M, :self.orig_N], cmap='gray')
        axes[1].set_title('Denoised Image')
        axes[1].axis('off')
    
        plt.tight_layout()
        plt.show()
        
    def plot_compressed_images(self, images, counts, compression_ratios):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    
        # Plot images
        for i, (ax, img, count) in enumerate(zip(axes.flat, images, counts)):
            ax.imshow(img[:self.orig_M, :self.orig_N], cmap='gray')
            compression_percent = compression_ratios[i] * 100
                
            ax.set_title(f'Compression: {compression_percent:.1f}%\nNon-zero coeffs: {count}')
            ax.axis('off')
    
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default="1")
    parser.add_argument('-i', '--image', default="moonlanding.png")
    
    args = parser.parse_args()
    fft_instance = FFT_Implementation()
    fft = FFT_Program(args.mode, args.image, fft_instance)