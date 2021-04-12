import cv2
import numpy as np
import matplotlib.pyplot as plt
# Kazım Muhammet Temiz 17050111021
# Mert Rıza Karadeniz 17050111054
# Nevzat Buğrahan Türk 17050111036

def inverse_filtering(image, H):
    fft_image = np.fft.fft2(image)
    fft_H = np.fft.fft2(H)
    h_shifted = np.fft.fftshift(fft_H)

    # Divide fft of degrading function to fft of original image
    fft_deconv = fft_image / h_shifted
    deconv = np.fft.ifftshift(np.fft.ifft2(fft_deconv))

    return np.abs(deconv)


def returnAtmosphericTurbulance(shape, k=0.0020, M=480, N=480):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Using the atmospheric turbulence degrading function formula from Gonzales Book
    for u in range(0, P):
        for v in range(0, Q):
            H[u, v] = np.exp(-k * ((u - M / 2) ** 2 + (v - N / 2) ** 2) ** 5 / 6)
    return H


img = cv2.imread('question_3.tif', 0)

img_shape = img.shape

H = returnAtmosphericTurbulance(img_shape)

result = inverse_filtering(img, H)

display = plt.figure()
display.add_subplot(1, 2, 1)
plt.imshow(img, "gray"), plt.title("Original")
display.add_subplot(1, 2, 2)
plt.imshow(result, "gray"), plt.title("Inverse Filtered Image")
plt.show(block=True)
