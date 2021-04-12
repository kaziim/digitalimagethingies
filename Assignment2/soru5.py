import cv2
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import numpy as np
# Kazım Muhammet Temiz 17050111021
# Mert Rıza Karadeniz 17050111054
# Nevzat Buğrahan Türk 17050111036

def CLSF(image, H, kernelised, gama=0.001):
    pad_fft = np.fft.fft2(kernelised)

    fft_image = np.fft.fft2(image)
    fft_H = np.fft.fft2(H)

    fft_deconv = (np.abs(fft_H) ** 2 / (np.abs(fft_H) ** 2 + gama * np.abs(pad_fft) ** 2)) * fft_image
    deconv = np.fft.ifft2(np.fft.fftshift(fft_deconv))

    return np.abs(deconv)


def returnAtmosphericTurbulance(shape, k=0.0025, M=480, N=480):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Using the atmospheric turbulence degrading function formula from Gonzales Book
    for u in range(0, P):
        for v in range(0, Q):
            H[u, v] = np.exp(-k * ((u - M / 2) ** 2 + (v - N / 2) ** 2) ** 5 / 6)
    return H


def convolve(source_image, kernel):  # Same function from HW2 Laplacian Filter
    (iH, iW) = source_image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW - 1) // 2
    source_image = cv2.copyMakeBorder(source_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype=float)

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            pixels = source_image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (pixels * kernel).sum()
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


img = cv2.imread("question_5.tif", 0)
img_shape = img.shape

kernelP = np.array((
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]))

H = returnAtmosphericTurbulance(img_shape)
kernelised = convolve(H, kernelP)
CLSF_filtered = CLSF(img, H, kernelised)

display = plt.figure()
display.add_subplot(1, 2, 1)
plt.imshow(img, "gray"), plt.title("Original")
display.add_subplot(1, 2, 2)
plt.imshow(CLSF_filtered, "gray"), plt.title("CLSF Filtered Image")
plt.show(block=True)
