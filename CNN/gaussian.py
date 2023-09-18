'''
Basic CNN implementation that will blur the image. default kernel works only for small images.

TODO: parallelize for loop for different color channels

'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def splitter(pixelArray):
    rPixel = np.array(pixelArray[:, :, 0])
    gPixel = np.array(pixelArray[:, :, 1])
    bPixel = np.array(pixelArray[:, :, 2])  # Corrected the color channel
    rOut = convolution(rPixel)
    gOut = convolution(gPixel)
    bOut = convolution(bPixel)
    finalOut = np.stack(
        (rOut, gOut, bOut), axis=-1).astype(np.uint8)  # Convert to 8-bit integer format

    plt.figure(figsize=(10, 5))  # Set the figure size

    # Original Image
    plt.subplot(1, 2, 1)  # Corrected the subplot position
    plt.imshow(pixelArray)
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis labels

    # Blurred Image
    plt.subplot(1, 2, 2)  # Corrected the subplot position
    plt.imshow(finalOut)
    plt.title("Blurred Image")
    plt.axis('off')  # Turn off axis labels

    plt.tight_layout()  # Adjust layout spacing
    plt.show()

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = 255 * (image - min_val) / (max_val - min_val)
    return normalized_image

def convolution(input):
    output = np.zeros(input.shape)
    kernel = np.array([[1.96519161e-05, 2.39409349e-04, 1.07295826e-03, 1.76900911e-03,
        1.07295826e-03, 2.39409349e-04, 1.96519161e-05],
       [2.39409349e-04, 2.91660295e-03, 1.30713076e-02, 2.15509428e-02,
        1.30713076e-02, 2.91660295e-03, 2.39409349e-04],
       [1.07295826e-03, 1.30713076e-02, 5.85815363e-02, 9.65846250e-02,
        5.85815363e-02, 1.30713076e-02, 1.07295826e-03],
       [1.76900911e-03, 2.15509428e-02, 9.65846250e-02, 1.59241126e-01,
        9.65846250e-02, 2.15509428e-02, 1.76900911e-03],
       [1.07295826e-03, 1.30713076e-02, 5.85815363e-02, 9.65846250e-02,
        5.85815363e-02, 1.30713076e-02, 1.07295826e-03],
       [2.39409349e-04, 2.91660295e-03, 1.30713076e-02, 2.15509428e-02,
        1.30713076e-02, 2.91660295e-03, 2.39409349e-04],
       [1.96519161e-05, 2.39409349e-04, 1.07295826e-03, 1.76900911e-03,
        1.07295826e-03, 2.39409349e-04, 1.96519161e-05]])  # Averaging kernel for blurring

    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2

    padded_input = np.pad(input, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    for i in range(pad_height, padded_input.shape[0] - pad_height):
        for j in range(pad_width, padded_input.shape[1] - pad_width):
            region = padded_input[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]
            output[i-pad_height, j-pad_width] = np.sum(region * kernel)

    return normalize_image(output)

def fileHandling():
    directory_path = "../CNN/Assets/image"
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
fileDir = "./Assets/image/"+fileName
img = Image.open(fileDir)

pixedData = list(img.getdata())
pixelArray = np.array(pixedData)

w, h = img.size
pixelArray = pixelArray.reshape(h, w, 3)

splitter(pixelArray)
