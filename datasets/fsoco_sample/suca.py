import json
import cv2
import matplotlib.pyplot as plt

# Load the image (make sure it's in the same directory or provide the full path)
image = cv2.imread("images/ff_00258.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the JSON annotations
with open("bounding_boxes/ff_00258.png.json", "r") as f:
    data = json.load(f)

# Draw bounding boxes
for obj in data["objects"]:
    (x1, y1), (x2, y2) = obj["points"]["exterior"]
    label = obj["classTitle"]
    color = (255, 255, 0) if "yellow" in label else (0, 0, 255)  # Yellow or Blue
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Show the image
plt.figure(figsize=(16, 10))
plt.imshow(image)
plt.axis('off')
plt.title("Bounding Box Visualization")
plt.show()
