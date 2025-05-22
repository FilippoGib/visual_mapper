import cv2
import numpy as np
import matplotlib.pyplot as plt

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked: {x}, {y}")
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Cones", img_display)

def backproject_pixel_to_ground(u, v, K, D, R, t):
    # undistort pixel
    pts = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, K, D, P=None)  # returns normalized coordinates
    x_n, y_n = undistorted[0, 0]

    ray_cam = np.array([x_n, y_n, 1.0]) # homogeneous coordinates

    n_world = np.array([0, 0, 1])
    d_world = 0

    n_cam = R.T @ n_world
    d_cam = n_world @ t

    lam = -d_cam / (n_cam @ ray_cam)
    X_cam = lam * ray_cam

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

distortion = np.array([-0.07456768657066797, 0.058448908740040065, -0.0005864815992339291, -9.326095667942688e-05, -0.02495779671549341 ]) # k1, k2, p1, p2, k3 (opencv convention)

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

img = cv2.imread('image2.png')
img_display = img.copy()

cv2.imshow("Select Cones", img_display)
cv2.setMouseCallback("Select Cones", click_event)

print("Click on cone bases. Press 'q' to finish.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # press "q" when you are done
        break

cv2.destroyAllWindows()

real_world_points = np.empty((0, 3))

print("\nProjected cone positions in global frame (meters):")
for (u, v) in clicked_points:
    X_world = backproject_pixel_to_ground(u, v, K=intrinsics, D=distortion, R=rotation, t=translation)
    real_world_points = np.vstack([real_world_points, X_world])
    print(f"Pixel ({u}, {v}) -> World: {X_world}")

X = real_world_points[:, 0]  # forward (rigth hand convention for automotive)
Y = real_world_points[:, 1]  # lateral (rigth hand convention for automotive)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, c='red', s=100, label='Cones')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

plt.title('Cone positions in world (car) frame')
plt.xlabel('X (meters, forward)')
plt.ylabel('Y (meters, lateral)')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()





