import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy import stats

class FFT_Implementation:
    def __init__(self):
        pass
    
    def fft_2d(self, image_array) -> np.ndarray:
        result = np.zeros_like(image_array, dtype=complex)
        
        for i in range(image_array.shape[0]):
            result[i, :] = self.cooley_tukey_fft(image_array[i].astype(float))
        for j in range(image_array.shape[1]):
            result[:, j] = self.cooley_tukey_fft(result[:, j])
        
        return result
    
    def naive_fft(self, signal) -> np.ndarray:
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
        
        # Base case: If N is small, use the naive FFT
        if N <= threshold:
            return self.naive_fft(signal)
        
        # Recursively apply FFT on even and odd indices
        even_part = self.cooley_tukey_fft(signal[::2], threshold)
        odd_part = self.cooley_tukey_fft(signal[1::2], threshold)
        
        factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        combined = np.zeros(N, dtype=complex)
        for k in range(N // 2):
            combined[k] = even_part[k] + factor[k] * odd_part[k]
            combined[k + N // 2] = even_part[k] - factor[k] * odd_part[k]
        
        return combined
    
    def inverse_fft_2d(self, freq_array) -> np.ndarray:
        result = np.zeros_like(freq_array, dtype=complex)
        M, N = freq_array.shape
        
        # Apply 1D inverse FFT to rows
        for i in range(M):
            result[i, :] = self.cooley_tukey_inverse_fft(freq_array[i])
            
        # Apply 1D inverse FFT to columns of the result
        for j in range(N):
            result[:, j] = self.cooley_tukey_inverse_fft(result[:, j])
        
        return result
    
    def naive_inverse_fft(self, signal) -> np.ndarray:
        N = len(signal)
        output = np.zeros(N, dtype=complex)
        
        for k in range(N):
            sum_val = 0j
            for n in range(N):
                angle = (2 * np.pi * k * n) / N  # Positive angle for inverse
                sum_val += signal[n] * (np.cos(angle) + 1j * np.sin(angle))
            output[k] = sum_val / N  # Apply 1/N scaling factor
        
        return output
    
    def cooley_tukey_inverse_fft(self, signal: np.ndarray, threshold: int = 8) -> np.ndarray:
        N = len(signal)
        
        # Base case: If N is small, use the naive inverse FFT
        if N <= threshold:
            return self.naive_inverse_fft(signal)
        
        # Split into even and odd parts
        even_part = self.cooley_tukey_inverse_fft(signal[::2], threshold)
        odd_part = self.cooley_tukey_inverse_fft(signal[1::2], threshold)
        
        # Compute twiddle factors with positive exponent for inverse
        k = np.arange(N // 2)
        factor = np.exp(2j * np.pi * k / N)  # Note the positive exponent
        
        # Combine results
        combined = np.zeros(N, dtype=complex)
        combined[:N//2] = even_part + factor * odd_part
        combined[N//2:] = even_part - factor * odd_part
        
        # Apply scaling factor at each level
        return combined / 2  # Distribute the N scaling factor across log(N) levels

    def shift_zero_frequency(self, fft_result: np.ndarray) -> np.ndarray:
        M, N = fft_result.shape
        shifted = np.zeros_like(fft_result)
        
        for i in range(M):
            for j in range(N):
                new_i = (i + M//2) % M
                new_j = (j + N//2) % N
                shifted[new_i, new_j] = fft_result[i, j]
                
        return shifted
    
    def shift_zero_frequency_back(self, fft_result: np.ndarray) -> np.ndarray:
        M, N = fft_result.shape
        shifted_back = np.zeros_like(fft_result)
        
        for i in range(M):
            for j in range(N):
                new_i = (i - M//2) % M
                new_j = (j - N//2) % N
                shifted_back[new_i, new_j] = fft_result[i, j]
            
        return shifted_back
    
    def denoise(self, image_array: np.ndarray, cutoff_ratio: float = 0.2) -> None:
        fft_result = self.shift_zero_frequency(self.fft_2d(image_array))
        
        # Create frequency mask
        M, N = fft_result.shape
        mask = np.zeros_like(fft_result)
        cutoff_x = int(M * cutoff_ratio / 2)
        cutoff_y = int(N * cutoff_ratio / 2)
        mask[M//2 - cutoff_x : M//2 + cutoff_x, N//2 - cutoff_y : N//2 + cutoff_y] = 1
        
        # Apply mask and shift back
        filtered_fft = self.shift_zero_frequency_back(fft_result * mask)
        
        # Compute inverse FFT
        denoised = self.inverse_fft_2d(filtered_fft).real
        
        non_zero_count = np.count_nonzero(filtered_fft)
        total_coefficients = M * N
        fraction_retained = non_zero_count / total_coefficients
        print(f"Non-zero coefficients retained: {non_zero_count}")
        print(f"Fraction of original Fourier coefficients retained: {fraction_retained:.4f}")
        
        return denoised
    
    def compress_threshold(self, image_array: np.ndarray, compression_ratios: list = [0, 0.1, 0.3, 0.5, 0.7, 0.99]) -> list:
        # Compute FFT and shift to center
        fft_result = self.fft_2d(image_array)
        shifted_fft = self.shift_zero_frequency(fft_result)
        
        # Store results
        compressed_images = []
        nonzero_counts = []
        
        # Get magnitudes for thresholding
        magnitudes = np.abs(shifted_fft)
        total_coeffs = np.prod(magnitudes.shape)
        
        for ratio in compression_ratios:
            if ratio == 0:
                # No compression case
                compressed_images.append(image_array)
                nonzero_counts.append(total_coeffs)
                continue
                
            # Calculate threshold value to keep desired percentage of coefficients
            threshold = np.percentile(magnitudes, ratio * 100)
            
            # Create mask for coefficients above threshold
            mask = magnitudes > threshold
            filtered_fft = shifted_fft * mask
            
            # Count non-zero coefficients
            nonzero_count = np.count_nonzero(mask)
            nonzero_counts.append(nonzero_count)
            
            # Inverse FFT to get compressed image
            unshifted = self.shift_zero_frequency_back(filtered_fft)
            compressed = self.inverse_fft_2d(unshifted).real
            compressed_images.append(compressed)
            
        return compressed_images, nonzero_counts
    
    def compress_frequency_based(self, image_array: np.ndarray, low_freq_ratio: float = 0.1, 
                               high_freq_ratios: list = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]) -> list:
        # Compute FFT and shift to center
        fft_result = self.fft_2d(image_array)
        shifted_fft = self.shift_zero_frequency(fft_result)
        
        M, N = shifted_fft.shape
        compressed_images = []
        nonzero_counts = []
        
        # Create low frequency mask
        low_freq_mask = np.zeros_like(shifted_fft)
        cutoff_x = int(M * low_freq_ratio / 2)
        cutoff_y = int(N * low_freq_ratio / 2)
        low_freq_mask[M//2 - cutoff_x : M//2 + cutoff_x, 
                     N//2 - cutoff_y : N//2 + cutoff_y] = 1
        
        # Get high frequency coefficients
        high_freq_coeffs = shifted_fft * (1 - low_freq_mask)
        high_freq_magnitudes = np.abs(high_freq_coeffs)
        
        for ratio in high_freq_ratios:
            if ratio == 1.0:
                # No compression case
                compressed_images.append(image_array)
                nonzero_counts.append(np.prod(shifted_fft.shape))
                continue
            
            # Keep low frequencies and threshold high frequencies
            if ratio > 0:
                threshold = np.percentile(high_freq_magnitudes[high_freq_magnitudes > 0], 
                                       (1 - ratio) * 100)
                high_freq_mask = high_freq_magnitudes > threshold
            else:
                high_freq_mask = np.zeros_like(shifted_fft)
            
            # Combine low and high frequency components
            filtered_fft = (shifted_fft * low_freq_mask + 
                          shifted_fft * (1 - low_freq_mask) * high_freq_mask)
            
            # Count non-zero coefficients
            nonzero_count = np.count_nonzero(filtered_fft)
            nonzero_counts.append(nonzero_count)
            
            # Inverse FFT to get compressed image
            unshifted = self.shift_zero_frequency_back(filtered_fft)
            compressed = self.inverse_fft_2d(unshifted).real
            compressed_images.append(compressed)
            
        return compressed_images, nonzero_counts
    
    def benchmark_performance(self, 
                            min_power: int = 1, 
                            max_power: int = 8, 
                            num_trials: int = 10,
                            confidence_level: float = 0.95) -> None:
        powers = range(min_power, max_power + 1)
        sizes = [2**p for p in powers]
        
        # Arrays to store results
        naive_times = []
        fft_times = []
        naive_stds = []
        fft_stds = []
        
        print("\nBenchmarking Performance:")
        print("------------------------")
        print(f"Running {num_trials} trials for each size")
        print(f"Confidence level: {confidence_level*100}%")
        
        for size in sizes:
            print(f"\nTesting size {size}x{size}:")
            
            # Arrays to store times for this size
            naive_trials = []
            fft_trials = []
            
            for trial in range(num_trials):
                # Generate random 2D array
                test_array = np.random.rand(size, size)
                
                # Time naive implementation
                start_time = perf_counter()
                for i in range(size):
                    _ = self.naive_fft(test_array[i])
                for j in range(size):
                    _ = self.naive_fft(test_array[:, j])
                naive_trials.append(perf_counter() - start_time)
                
                # Time Cooley-Tukey implementation
                start_time = perf_counter()
                _ = self.fft_2d(test_array)
                fft_trials.append(perf_counter() - start_time)
                
                print(f"  Trial {trial + 1}/{num_trials} complete", end='\r')
            
            # Calculate statistics for this size
            naive_mean = np.mean(naive_trials)
            naive_std = np.std(naive_trials, ddof=1)
            naive_times.append(naive_mean)
            naive_stds.append(naive_std)
            print(f"Naive FFT - Mean: {naive_mean:.4f}s, Std: {naive_std:.4f}s")
            
            fft_mean = np.mean(fft_trials)
            fft_std = np.std(fft_trials, ddof=1)
            fft_times.append(fft_mean)
            fft_stds.append(fft_std)
            print(f"Cooley-Tukey FFT - Mean: {fft_mean:.4f}s, Std: {fft_std:.4f}s")
        
        # Calculate confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot Cooley-Tukey FFT
        plt.errorbar(sizes, fft_times, 
                    yerr=np.array(fft_stds) * z_score,
                    fmt='o-', label='Cooley-Tukey FFT',
                    capsize=5, capthick=1, elinewidth=1,
                    color='blue', markersize=8)
        
        plt.errorbar(sizes[:len(naive_times)], naive_times,
                    yerr=np.array(naive_stds) * z_score,
                    fmt='o-', label='Naive FFT',
                    capsize=5, capthick=1, elinewidth=1,
                    color='red', markersize=8)
        
        # Add theoretical complexity lines
        x_theory = np.array(sizes)
        # Normalize to match the last point of FFT timing
        fft_theory = (x_theory * np.log2(x_theory)) * (fft_times[-1] / (sizes[-1] * np.log2(sizes[-1])))
        plt.plot(x_theory, fft_theory, '--', color='lightblue', 
                label='O(N log N) theory', alpha=0.5)
        
        naive_theory = (x_theory[:len(naive_times)] ** 2) * (naive_times[-1] / (sizes[len(naive_times)-1] ** 2))
        plt.plot(x_theory[:len(naive_times)], naive_theory, '--', 
                color='pink', label='O(N²) theory', alpha=0.5)
        
        # Customize plot
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Array Size (N×N)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('FFT Performance Comparison\n'
                 f'with {confidence_level*100}% Confidence Intervals\n'
                 f'({num_trials} trials per size)')
        plt.legend()
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("------------------")
        print("Cooley-Tukey FFT:")
        for size, mean, std in zip(sizes, fft_times, fft_stds):
            print(f"Size {size}x{size}:")
            print(f"  Mean: {mean:.4f}s")
            print(f"  Std Dev: {std:.4f}s")
            print(f"  {confidence_level*100}% CI: [{mean - z_score*std:.4f}, {mean + z_score*std:.4f}]")
        
        if naive_times:
            print("\nNaive FFT:")
            for size, mean, std in zip(sizes[:len(naive_times)], naive_times, naive_stds):
                print(f"Size {size}x{size}:")
                print(f"  Mean: {mean:.4f}s")
                print(f"  Std Dev: {std:.4f}s")
                print(f"  {confidence_level*100}% CI: [{mean - z_score*std:.4f}, {mean + z_score*std:.4f}]")
        
        plt.tight_layout()
        plt.show()
    
    
        