import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_attention_map(attention_map: np.ndarray, image_path: str, result_path: str):
    _, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes[0].matshow(attention_map.squeeze())
    img = cv2.imread(image_path)
    attention_map = cv2.resize(attention_map, (img.shape[1], img.shape[0]))
    attention_map = np.uint8(255 * attention_map)
    attention_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    result_img = (attention_map * 0.4 + img).astype(np.float32)
    axes[1].imshow(img[..., ::-1])
    axes[2].imshow((result_img / 255)[..., ::-1])
    axes[0].axis("off")
    axes[1].axis("off")
    axes[2].axis("off")
    plt.savefig(result_path)
