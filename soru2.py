from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Kazım Muhammet Temiz 17050111021
#Mert Rıza Karadeniz 17050111054
#Nevzat Buğrahan Türk 17050111036

def convolve(source_image, kernel):
    # Kaynak resmin ve matrix kernelin spatial çözünürlüğünü al
    (iH, iW) = source_image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # Çıktı resim için yer açıyoruz ve kenarları padliyoruz
    # output bizim resmimizle aynı çözünürlüğe sahip ama her pixel 0 grayscale
    pad = (kW - 1) // 2
    source_image = cv2.copyMakeBorder(source_image, pad, pad, pad, pad,
                                      cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype=float)

    # matrixi her pikselle çarp ve topla
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # resimde matrixi çarpacağımız 3x3 alanı/pixelleri seçiyoruz
            # 0 olma ihtimaline karşı +1 eklendi
            pixels = source_image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # seçtiğimiz pixelleri kernel matriximizle çarpıyoruz
            k = (pixels * kernel).sum()

            # Output resmin aynı koordinatlı yerine bulduğumuz değeri yazıyoruz
            output[y - pad, x - pad] = k

    # Output resimdeki değerler 255i geçebileceğinden rescale ediyoruz
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


sharpen = np.array((
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]))

img = cv2.imread("question_2.tif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Applying spatial sharpening filter")
Result = convolve(gray, sharpen)

display = plt.figure()
display.add_subplot(1,2, 1)
plt.imshow(img,"gray"), plt.title("Original")
display.add_subplot(1,2, 2)
plt.imshow(Result,"gray"), plt.title("Spatial Sharpening Filter")
plt.show(block=True)


