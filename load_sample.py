import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

root_path = os.getcwd()
test_dataset_path=root_path+"/test_data/"
image_path=test_dataset_path+"test_images.npy"
label_path=test_dataset_path+"test_labels.npy"

test_img_load = list(np.load(image_path))
test_lbl_load = list(np.load(label_path))

# Extract True / False labels
test_lbl_0 = [i for i, lbl in enumerate(test_lbl_load) if lbl == 0]
test_lbl_1 = [i for i, lbl in enumerate(test_lbl_load) if lbl == 1]

index = test_lbl_0[10]

# Extract a sample image
sample_img = test_img_load[index]
sample_img_rgb = sample_img[:,:,0:3]
sample_img_ir = sample_img[:,:,3]

plt.imshow(sample_img_rgb)
plt.savefig('0_rgb.png')

plt.imshow(sample_img_ir, cmap='gray')
plt.savefig('0_ir.png')