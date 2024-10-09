import cv2

img = cv2.imread("/root/3d_model/1_512_512.jpg")

def crop_black(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(grayscale, 32, 255, cv2.THRESH_BINARY)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    foreground = img[y:y+h, x:x+w]
    return foreground
foreground =crop_black(img)
cv2.imwrite("foreground.png", foreground)