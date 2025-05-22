import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon, Rectangle

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked: {x}, {y}")
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Cones", img_display)

def backproject_pixel_to_ground(u, v, K, R, t):

    # create a direction from the pixel
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    ray_cam = np.array([x_n, y_n, 1.0])

    # define the z=0 plane in the world reference frame
    n_world = np.array([0, 0, 1]) # versor
    d_world = 0 # height in the direction of the versor

    # convert plane in camera frame
    n_cam = R.T @ n_world
    d_cam = n_world @ t  # = t[2]

    # find lambda = length of the ray that starts from center of the camera and intersects the plane
    lam = -d_cam / (n_cam @ ray_cam)

    # get exact point (in camera frame)
    X_cam = lam * ray_cam

    # bring point to world frame
    X_world = R @ X_cam + t
    return X_world


# calibration of HD camera see file "SN<...>.conf"
intrinsics = np.array([
    [530.487, 0.0, 648.425],
    [0.0, 530.603, 365.797],
    [0.0,     0.0,     1.0]
])


h = 1.0

translation = np.array([0.0, 0.0, h])

# assumes zero roll, pitch, yaw
rotation = np.array([ # just swapping axes around
    [0,  0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])

# accounts for 20 degrees of downwards rotation expressed in the camera frame (idk if it is the right way to do it)
theta_deg = -20.0
theta_rad = np.deg2rad(theta_deg)

R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta_rad), -np.sin(theta_rad)],
    [0, np.sin(theta_rad),  np.cos(theta_rad)]
])

rotation = rotation @ R_x # apply pitch to original rotation matrix

############################### MANUALLY SELECT PIXELS OF CONES BASE ###############################
clicked_points = []
crosshair_color = (255, 255, 255)
marker_color = (0, 165, 255)  # Orange-like color
font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread('image2.png')
img_display = img.copy()

# Create a copy for drawing cursor overlays separately
overlay = img.copy()

# Initialize mouse position
mouse_pos = (0, 0)

def click_event(event, x, y, flags, param):
    global img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

        # Elegant thin cross: longer horizontal, shorter vertical
        horizontal_len = 6
        vertical_len = 3

        cv2.line(img_display, (x - horizontal_len, y), (x + horizontal_len, y), marker_color, 1)
        cv2.line(img_display, (x, y - vertical_len), (x, y + vertical_len), marker_color, 1)

        # Number label near the point
        label_pos = (x + 10, y - 10)
        label = str(len(clicked_points))
        
        # White border (drawn first, slightly thicker)
        cv2.putText(img_display, label, label_pos, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Black fill (drawn second, thinner)
        cv2.putText(img_display, label, label_pos, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        global overlay, mouse_pos
        mouse_pos = (x, y)
        overlay = img_display.copy()

        # Draw crosshairs at the cursor
        cv2.line(overlay, (x, 0), (x, overlay.shape[0]), crosshair_color, 1)
        cv2.line(overlay, (0, y), (overlay.shape[1], y), crosshair_color, 1)

        # Optional: show coordinates at top-left
        cv2.rectangle(overlay, (10, 10), (140, 35), (0, 0, 0), -1)
        cv2.putText(overlay, f"({x}, {y})", (15, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

cv2.namedWindow("Select Cones")
cv2.setMouseCallback("Select Cones", click_event)

while True:
    cv2.imshow("Select Cones", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Backproject to 3D world
real_world_points = np.empty((0, 3))

for i, (u, v) in enumerate(clicked_points):
    X_world = backproject_pixel_to_ground(u, v, K=intrinsics, R=rotation, t=translation)
    real_world_points = np.vstack([real_world_points, X_world])
    print(f"{i+1:02d}. Pixel ({u}, {v}) -> World: {X_world}")

X = real_world_points[:, 0]  # forward (right-hand convention)
Y = real_world_points[:, 1]  # lateral (right-hand convention)

############################### VISUALIZATION ###############################

plt.figure(figsize=(10, 6))
ax = plt.gca()

# Parameters for cone shape
cone_height = 0.6   # meters
cone_width = 0.25   # meters
stripe_height = 0.20  # height of the white stripe
stripe_offset = 0.30  # vertical offset from tip for stripe

# Add triangle patches for each cone
for i, (x, y) in enumerate(zip(X, Y)):
    # Triangle points for cone (isosceles triangle)
    triangle = np.array([
        [y, x],                                # tip
        [y - cone_width / 2, x - cone_height],  # bottom left
        [y + cone_width / 2, x - cone_height]   # bottom right
    ])
    cone_patch = Polygon(triangle, closed=True, facecolor='blue', edgecolor='black', zorder=3)
    ax.add_patch(cone_patch)

    # Add the cone index label above the cone
    plt.text(
        y + cone_width / 2 - 0.30,          # a bit to the right of the base
        x - cone_height,             # aligned with base + small vertical offset
        str(i + 1),
        fontsize=12,
        fontweight='bold',
        color='black',
        verticalalignment='bottom',
        zorder=5
    )

# Axes lines
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# Labels and title
plt.title('Cone Positions in Vehicle Frame', fontsize=16, fontweight='bold')
plt.xlabel('Lateral Position Y (m)', fontsize=14)
plt.ylabel('Forward Position X (m)', fontsize=14)

# Grid and tick marks
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.gca().xaxis.set_major_locator(MultipleLocator(2.0))
plt.gca().yaxis.set_major_locator(MultipleLocator(2.0))

# Maintain aspect ratio and invert X to show left as positive
plt.axis('equal')
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()





