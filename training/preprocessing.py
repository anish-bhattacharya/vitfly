import cv2

# Read the image
image = cv2.imread("/home/prasad/catkin_ws_vitfly/src/vitfly/training/datasets/data/170692293306/1706922932.608.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Display the original and edge-detected images
cv2.imwrite("original.png", image)
cv2.imwrite("edges.png", edges)
print("Saved images")

cv2.waitKey(0)
cv2.destroyAllWindows()