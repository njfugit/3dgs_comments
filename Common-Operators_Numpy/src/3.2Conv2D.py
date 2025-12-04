import numpy as np
import cv2

def convolution2D(image, kernel, strides=1, padding=0):
    # Check for multi-channel image
    if image.ndim == 3:
        img_h, img_w, img_c = image.shape
    else:
        img_h, img_w = image.shape
        image = np.expand_dims(image, axis=-1)
        img_c = 1

    # Check kernel dimensions
    if kernel.ndim == 2:
        # If kernel is 2D, expand it to be applied to all channels
        kernel = np.expand_dims(kernel, axis=-1)
        print("aa  kernel shape:",kernel.shape)
    
    if kernel.shape[-1] != img_c and kernel.shape[-1] != 1:
        raise ValueError("Kernel channels must be 1 or match the number of image channels.")

    kernel_h, kernel_w, kernel_c = kernel.shape

    if padding > 0:
        image_pad = np.pad(image, pad_width=((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    else:
        image_pad = image
        
    padded_h, padded_w, _ = image_pad.shape
    out_h = (padded_h - kernel_h) // strides + 1
    out_w = (padded_w - kernel_w) // strides + 1
    
    output = np.zeros((out_h, out_w))
    
    for y in range(out_h):
        for x in range(out_w):
            start_y = y * strides
            start_x = x * strides
            roi = image_pad[start_y:start_y+kernel_h, start_x:start_x+kernel_w, :]
            output[y, x] = np.sum(roi * kernel)
            
    return output

if __name__ == "__main__":
    # --- Single Channel Example ---
    print("--- Single Channel Test ---")
    input_image = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])

    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]) / 9.0

    print("--- Original Input ---")
    print(input_image)
    print("\n--- Kernel ---")
    print(kernel)

    output_valid = convolution2D(input_image, kernel, strides=1, padding=0)
    print("\n--- Result (padding=0, stride=1) ---")
    print(output_valid)

    output_same = convolution2D(input_image, kernel, strides=1, padding=1)
    print("\n--- Result (padding=1, stride=1) ---")
    print(output_same)

    output_stride2 = convolution2D(input_image, kernel, strides=2, padding=1)
    print("\n--- Result (padding=1, stride=2) ---")
    print(output_stride2)

    # --- Three Channel Example ---
    print("\n\n--- Three Channel Test ---")
    input_image_3ch = np.stack([
        input_image * 0.5,
        input_image,
        input_image * 1.5
    ], axis=-1)

    kernel_3ch = np.stack([
        kernel * 0.8,
        kernel,
        kernel * 1.2
    ], axis=-1)

    print("--- Original 3-channel input (shape: {})" .format(input_image_3ch.shape))
    print("\n--- 3-channel kernel (shape: {})" .format(kernel_3ch.shape))

    output_3ch = convolution2D(input_image_3ch, kernel_3ch, strides=1, padding=1)
    print("\n--- 3-channel result (padding=1, stride=1) (shape: {})" .format(output_3ch.shape))
    print(output_3ch)
    
    # --- Three Channel Example with 2D kernel ---
    print("\n\n--- Three Channel Test with 2D kernel ---")
    output_3ch_2d_kernel = convolution2D(input_image_3ch, kernel, strides=1, padding=1)
    print("\n--- 3-channel with 2d kernel result (padding=1, stride=1) (shape: {})" .format(output_3ch_2d_kernel.shape))
    print(output_3ch_2d_kernel)
