import cv2
import random
import numpy as np

def rotate_with_heatmap(image, heatmap, max_angle=20, p=0.5):
    if random.random() < p:
        angle = random.randint(-max_angle, max_angle)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image_cv2 = cv2.warpAffine(image, M, (w, h))

        # Rotate the heatmap using OpenCV
        (h, w) = heatmap.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_heatmap_cv2 = cv2.warpAffine(heatmap, M, (w, h))
        return rotated_image_cv2, rotated_heatmap_cv2
    else:
        return image, heatmap
    
def hlfip_with_heatmap(image, heatmap, p=0.5):
    if random.random() < p:
        flip_image = cv2.flip(image, 1)
        flip_heatmap = cv2.flip(heatmap, 1)
        return flip_image, flip_heatmap
    else:
        return image, heatmap
    
if __name__ == '__main__':
    image = cv2.imread("/root/sagittal_model/images/sagittal_t1/4003253/1054713880/5_384_384.jpg")
    heat_map = np.zeros((12, 12))
    heat_map[4, 3] = 1
    heat_map[5, 3] = 1
    print(heat_map)
    cv2.imwrite("raw_heat_map.png", heat_map)
    rotated_image_cv2, rotated_heatmap_cv2 = hlfip_with_heatmap(image, heat_map, p=1)
    cv2.imwrite("rotate_image.png", rotated_image_cv2)
    cv2.imwrite("rotate_heat_map.png", rotated_heatmap_cv2)
    print(rotated_heatmap_cv2)

