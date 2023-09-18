from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from fileHandling import fileHandling
def convolve_channel(args):
    channel, kernel = args
    i_h, i_w = channel.shape
    k_h, k_w = kernel.shape

    # Calculate padding
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Pad the image channel
    padded_channel = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Prepare the output array for the channel
    output = np.zeros_like(channel)

    # Apply convolution
    for i in range(i_h):
        for j in range(i_w):
            output[i, j] = np.sum(padded_channel[i: i + k_h, j: j + k_w] * kernel)

    return output

def main():
    fileDir = fileHandling("./Assets/image/")
    print(fileDir)
    image = Image.open(fileDir)
    image_np = np.array(image)

    # Define a random kernel
    kernel_size = 3
    random_kernel = np.random.randn(kernel_size, kernel_size)
    print(random_kernel)
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=3) as pool:  # We use 3 processes for the 3 channels
        # Apply the random kernel to the image using multiprocessing
        channels = [image_np[:, :, i] for i in range(3)]
        results = pool.map(convolve_channel, [(channel, random_kernel) for channel in channels])

    # Combine the processed channels to get the convolved image
    convolved_image_np = np.stack(results, axis=-1)

    # Ensure pixel values are within [0, 255] range after convolution
    convolved_image_np = np.clip(convolved_image_np, 0, 255)

    # Display images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.uint8(convolved_image_np))
    plt.title('Convolved Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
