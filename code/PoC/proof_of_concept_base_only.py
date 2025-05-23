import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon

ACCOUNT_FOR_DISTORTION = True

def click_event(event, x, y, flags, param):
    state = param  # param is a dict holding shared state
    if event == cv2.EVENT_LBUTTONDOWN:
        state['clicked_points'].append((x, y))

        horizontal_len = 6
        vertical_len = 3

        cv2.line(state['img_display'], (x - horizontal_len, y), (x + horizontal_len, y), state['marker_color'], 1)
        cv2.line(state['img_display'], (x, y - vertical_len), (x, y + vertical_len), state['marker_color'], 1)

        label_pos = (x + 10, y - 10)
        label = str(len(state['clicked_points']))

        cv2.putText(state['img_display'], label, label_pos, state['font'], 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(state['img_display'], label, label_pos, state['font'], 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    elif event == cv2.EVENT_MOUSEMOVE:
        state['mouse_pos'] = (x, y)
        state['overlay'] = state['img_display'].copy()

        cv2.line(state['overlay'], (x, 0), (x, state['overlay'].shape[0]), state['crosshair_color'], 1)
        cv2.line(state['overlay'], (0, y), (state['overlay'].shape[1], y), state['crosshair_color'], 1)

        cv2.rectangle(state['overlay'], (10, 10), (140, 35), (0, 0, 0), -1)
        cv2.putText(state['overlay'], f"({x}, {y})", (15, 30), state['font'], 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def backproject_pixel_to_ground_with_distortion(u, v, K, D, R, t):
    pts = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, K, D, P=None)
    x_n, y_n = undistorted[0, 0]
    ray_cam = np.array([x_n, y_n, 1.0])
    n_world = np.array([0, 0, 1])
    n_cam = R.T @ n_world
    d_cam = n_world @ t
    lam = -d_cam / (n_cam @ ray_cam)
    X_cam = lam * ray_cam
    X_world = R @ X_cam + t
    return X_world

def backproject_pixel_to_ground(u, v, K, R, t):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    ray_cam = np.array([x_n, y_n, 1.0])
    n_world = np.array([0, 0, 1])
    d_world = 0
    n_cam = R.T @ n_world
    d_cam = n_world @ t
    lam = -d_cam / (n_cam @ ray_cam)
    X_cam = lam * ray_cam
    X_world = R @ X_cam + t
    return X_world

def main():
    # Calibration
    intrinsics = np.array([
        [530.487, 0.0, 648.425],
        [0.0, 530.603, 365.797],
        [0.0,     0.0,     1.0]
    ])
    h = 1.0
    translation = np.array([0.0, 0.0, h])
    rotation = np.array([
        [0,  0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    distortion = np.array([-0.07456768657066797, 0.058448908740040065, -0.0005864815992339291, -9.326095667942688e-05, -0.02495779671549341 ])
    theta_deg = -17.0
    theta_rad = np.deg2rad(theta_deg)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), -np.sin(theta_rad)],
        [0, np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    rotation = rotation @ R_x

    # State for mouse callback
    state = {
        'clicked_points': [],
        'crosshair_color': (255, 255, 255),
        'marker_color': (0, 165, 255),
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'img_display': None,
        'overlay': None,
        'mouse_pos': (0, 0)
    }

    img = cv2.imread('image3.png')
    state['img_display'] = img.copy()
    state['overlay'] = img.copy()

    cv2.namedWindow("Select Cones")
    cv2.setMouseCallback("Select Cones", click_event, param=state)

    while True:
        cv2.imshow("Select Cones", state['overlay'])
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Backproject to 3D world
    real_world_points = np.empty((0, 3))
    for i, (u, v) in enumerate(state['clicked_points']):
        if ACCOUNT_FOR_DISTORTION:
            X_world = backproject_pixel_to_ground_with_distortion(u, v, K=intrinsics, D=distortion, R=rotation, t=translation)
        else:
            X_world = backproject_pixel_to_ground(u, v, K=intrinsics, R=rotation, t=translation)
        X_world = X_world.reshape(-1)
        real_world_points = np.vstack([real_world_points, X_world])
        formatted = ", ".join(f"{x:.2f}" for x in X_world)
        print(f"{i+1:02d}. Pixel ({u}, {v}) -> World: Cone_{i+1} = ({formatted})")

    X = real_world_points[:, 0]
    Y = real_world_points[:, 1]

    # Visualization
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    cone_height = 0.6
    cone_width = 0.25
    for i, (x, y) in enumerate(zip(X, Y)):
        base_left  = (y - cone_width/2, x)
        base_right = (y + cone_width/2, x)
        # tip set back by cone_height:
        tip        = (y,            x + cone_height)

        triangle = np.array([base_left, base_right, tip])
        cone_patch = Polygon(triangle, closed=True, facecolor='blue', edgecolor='black', zorder=3)
        ax.add_patch(cone_patch)
        plt.text(
            y + cone_width / 2 - 0.30,
            x,
            str(i + 1),
            fontsize=12,
            fontweight='bold',
            color='black',
            verticalalignment='bottom',
            zorder=5
        )
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Cone Positions in Vehicle Frame', fontsize=16, fontweight='bold')
    plt.xlabel('Lateral Position Y (m)', fontsize=14)
    plt.ylabel('Forward Position X (m)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.gca().xaxis.set_major_locator(MultipleLocator(2.0))
    plt.gca().yaxis.set_major_locator(MultipleLocator(2.0))
    plt.axis('equal')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
