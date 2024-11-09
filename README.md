Your application should be named fft.py, and invoked as follows:
python fft.py [-m mode] [-i image]

    - mode (optional):
        - [1] (Default) Fast mode: Convert image to FFT form and display
        - [2] Denoise: The image is denoised by applying an FFT, truncating 
        high frequencies and then displayed
        - [3] Compress: Compress image and plot.
        - [4] Plot runtime graphs for the report
        
    - image (optional): Filename of the image for the DFT (default: given image by prof)
"""

"""
Your program's output depends on the mode you are running.

1 - Perform FFT and output a one by two subplot of the original image and next to it
    its Fourier transform. Fourier transform should be log scaled (import LogNorm from
    matplotlib.colors to produce a logarithmic colormap)
    
2 - Output a one by two subplot. Include the original image next to its denoised 
    version. To denoise take the FFT of the image, and set all high frequencies to zero
    before inverting to get back the filtered original. Where you choose to draw the line
    between a "high" and "low" frequency is up to you to design and tune to get the best
    result.
    
For 2, the FFT plot produced goes from 0 to 2pi, so any frequency close to 0 
and 2pi can be considered low. The program should print in the command line, the 
number of non-zeros you are using and the fraction they represent of the original 
Fourier coeffiecients.

3 - First, take the FFT of the image to compress it. (check assignment doc for more
    details on how to do this)
    
4 - Produce plots that summarize the runtime complexity of your algorithms. Your
    code should print in the command line the means and variances of the runtime
    of your algorithms versus the problem size + more qualitative details (see 
    details in Report section)


PYTHON VERSION USED PYTHON3.96