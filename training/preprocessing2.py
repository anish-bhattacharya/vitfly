import cv2
import numpy as np

image = cv2.imread("/home/prasad/catkin_ws_vitfly/src/vitfly/training/datasets/data/170692293306/1706922932.608.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Convert edges (1-channel) to 3-channel so it matches original image shape
edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Combine images side by side (left + right)
combined =  cv2.hconcat([image, edges_color])

# Save results
cv2.imwrite("original.png", image)
cv2.imwrite("edges2.png", cv2.resize(image, (90, 60)))
cv2.imwrite("combined.png", combined)

print("Saved images")

cv2.waitKey(0)
cv2.destroyAllWindows()