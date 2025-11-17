import cv2
from matplotlib import pyplot as plt


def read_image(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Original image")
    plt.show()
    return image

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                 
    gray = cv2.GaussianBlur(gray, (3, 3), 0)                       
    edged = cv2.Canny(gray, 10, 250)                                
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))      
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closed)
    plt.show()
    
    return closed

def get_contours(binary_image):
    contours = cv2.findContours(binary_image.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        return "rectangle" 
    elif vertices > 6:
        return "circle"
    else:
        return "polygon"


def recognize_objects(image, contours, save_path, find_type):
    count = 0
    annotated = image.copy()

    for cnt in contours:
        shape = classify_shape(cnt)
        if shape == find_type:
            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 3)
            count += 1

    print(f"Знайдено {count} об’єкт(ів) типу '{find_type}'")
    cv2.imwrite(save_path, annotated)

    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Detected objects")
    plt.show()


if __name__ == "__main__":
    image = read_image("triangle.jpg")
    processed = preprocess(image)
    contours = get_contours(processed) 
    recognize_objects(image, contours, "triangle_res.jpg", find_type="triangle")

    image = read_image("rectangle.jpg")
    processed = preprocess(image)
    contours = get_contours(processed) 
    recognize_objects(image, contours, "rectangle_res.jpg", find_type="rectangle")

    image = read_image("circle.jpg")
    processed = preprocess(image)
    contours = get_contours(processed) 
    recognize_objects(image, contours, "circle_res.jpg", find_type="circle")
