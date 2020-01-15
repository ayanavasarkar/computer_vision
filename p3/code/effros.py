import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.feature_extraction import image as img_features


def grow_image(sample_image, image, image_size, window_size):

    unfilled = image_size - np.count_nonzero(image)
    max_err_threshold = 0.3

    gauss_mask = get_gauss_mask(window_size)
    patches = img_features.extract_patches_2d(sample_image, (window_size, window_size))
    win_size = window_size
    while unfilled:
        flag = False

        lst = list()
        half_window = window_size
        for i in range(half_w, image.shape[0] - half_w):
            for j in range(half_w, image.shape[1] - half_w):
                if image[i, j] != 0:
                    continue
                #pixel = Pixel(i, j)
                neighborhood = image[pixel.x - half_window:pixel.x + half_window + 1,
                               pixel.y - half_window:pixel.y + half_window + 1]
                pixel.neighbor_count = np.count_nonzero(neighborhood)
                if pixel.neighbor_count > 0:
                    lst.append(pixel)

        pixel_list = sorted(lst, key=lambda x: x.neighbor_count, reverse=True)

        for pixel in pixel_list:
            template = image[pixel.x - half_window:pixel.x + half_window + 1, pixel.y - half_window:pixel.y + half_window + 1]

            threshold = 0.1

            mask = template != 0
            weight = np.multiply(gauss_mask, mask)
            total_weight = np.sum(weight)

            if total_weight == 0:
                total_weight = 1

            SSD = np.sum(np.multiply((patches - template) ** 2, weight), axis=(1, 2)) / float(total_weight)

            min_err = min(SSD)
            candidates = []
            for i, err in enumerate(SSD):
                if err <= min_err * (1 + threshold):
                    candidate = Pixel(0, 0)
                    candidate.error = err
                    candidate.value = patches[i, win_size / 2, win_size / 2]
                    candidates.append(candidate)

            candidate = np.random.choice(candidates)
            if candidate.error <= max_err_threshold:
                image[pixel.x, pixel.y] = candidate.value
                flag = True
                unfilled -= 1

        if not flag:
            max_err_threshold = np.multiply(max_err_threshold, 1.1)

    return image


if __name__ == '__main__':

    window_size = 11
    half_w = window_size

    size = (70, 70)

    imgsize = (70 + half_w * 2, 70 + half_w * 2)

    img = mpimg.imread("english.jpg")

    img = np.divide((0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]), 255.0)

    target = np.zeros(imgsize)

    seed_size = 5
    rand_x = np.random.randint(0, img.shape[0] - seed_size)
    rand_y = np.random.randint(0, img.shape[1] - seed_size)

    seed = img[rand_x:rand_x + seed_size, rand_y:rand_y + seed_size]
    center_x = np.divide((imgsize[0] - seed_size),2)
    center_y = np.divide((imgsize[1] - seed_size),2)

    target[center_x:center_x + seed_size, center_y:center_y + seed_size] = seed
    target = grow_image(img, target, size[0] * size[1], window_size)
    mpimg.imsave('3_7.png', target[half_w:-half_w, half_w:-half_w], cmap=plt.get_cmap('gray'))