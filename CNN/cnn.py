'''
Basic CNN implementation that will blur the image. default kernel works only for small images.

TODO: parallelize for loop for different color channels

'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def splitter(pixelArray, kernel):
    rPixel = np.array(pixelArray[:, :, 0])
    gPixel = np.array(pixelArray[:, :, 1])
    bPixel = np.array(pixelArray[:, :, 2])
    rOut = convolution(rPixel, kernel)
    gOut = convolution(gPixel, kernel)
    bOut = convolution(bPixel, kernel)
    rOut_normalized = normalize_image(rOut)
    gOut_normalized = normalize_image(gOut)
    bOut_normalized = normalize_image(bOut)
    finalOut = np.stack(
        (rOut_normalized, gOut_normalized, bOut_normalized), axis=-1)

    plt.figure(figsize=(10, 5))  # Set the figure size

    # Original Image
    plt.subplot(1, 2, 1)  # Create a subplot
    plt.imshow(pixelArray)
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis labels

    # Blurred Image
    plt.subplot(1, 2, 2)  # Create another subplot
    plt.imshow(finalOut)
    plt.title("Blurred Image")
    plt.axis('off')  # Turn off axis labels

    plt.tight_layout()  # Adjust layout spacing
    plt.show()


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def convolution(input, kernel):
    h, w = kernel.shape
    pad_height = h // 2
    pad_width = w // 2
    padded_input = np.pad(input, ((pad_height, pad_height),
                          (pad_width, pad_width)), mode='constant')

    output = np.zeros(input.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(padded_input[i:i+h, j:j+w] * kernel)
    return output


def fileHandling():
    directory_path = "../CNN/image"
    file_list = os.listdir(directory_path)
    file_count = len(file_list)
    if file_count==0:
        print("Please paste your image in image folder")
        exit(1)
    elif file_count==1:
        file_name = file_list[0]
        return file_name
    elif file_count>1:
        fileName = input("Multiple images found. Please input the name of image: ")
        file_path = os.path.join(directory_path, fileName)
        if os.path.exists(file_path):
            print("File exists.")
            return fileName
        else:
            print("File not found")
            exit(1)

fileName = fileHandling()
fileDir = "./image/"+fileName
img = Image.open(fileDir)

pixedData = list(img.getdata())
pixelArray = np.array(pixedData)
blurKernel = np.array(
    [[2, 2, 2, 2, 2, 2, 2],
     [2, 2, 2, 2, 2, 2, 2],
     [2, 2, 2, 2, 4, 2, 2],
     [2, 2, 2, 2, 8, 2, 2],
     [2, 2, 2, 2, 4, 2, 2],
     [2, 2, 2, 2, 2, 2, 2],
     [2, 2, 2, 2, 2, 2, 2]])
w, h = img.size
pixelArray = pixelArray.reshape(h, w, 3)

splitter(pixelArray, blurKernel)

