import cv2
import matplotlib.pyplot as plt
import numpy as np
# Kazım Muhammet Temiz 17050111021
# Mert Rıza Karadeniz 17050111054
# Nevzat Buğrahan Türk 17050111036

def wiener_filter(image, H, K=0.0025):
    fft_image = np.fft.fft2(image)
    fft_H = np.fft.fft2(H)

    fft_deconv = (np.abs(fft_H) ** 2 / (np.abs(fft_H) ** 2 + K) * 1 / fft_H) * fft_image
    deconv = np.fft.ifftshift(np.fft.ifft2(fft_deconv))

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


img = cv2.imread("question_4.tif", 0)
img_shape = img.shape

H = returnAtmosphericTurbulance(img_shape)

wiener_filtered = wiener_filter(img, H)

display = plt.figure()
display.add_subplot(1, 2, 1)
plt.imshow(img, "gray"), plt.title("Original")
display.add_subplot(1, 2, 2)
plt.imshow(wiener_filtered, "gray"), plt.title("Wiener Filtered Image")
plt.show(block=True)
