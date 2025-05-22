#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class BirdseyeNode(Node):
    def __init__(self):
        super().__init__('birdseye_node')
        self.get_logger().info('Birdseye INITIALIZATIO STARTED')

        # Camera params TODO: find them
        fx, fy = 800.0, 800.0
        cx, cy = 640.0, 360.0
        skew = 0.0

        self.K = np.array([[fx, skew, cx],
                           [0,  fy,   cy],
                           [0,   0,    1]])
        
        self.distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # I assume the images I get are already rectified

        # Extrinsics: camera → car
        self.R = np.array( # if the camera is pointing perfectly forward
            [[ 0, 0, 1],
            [-1, 0, 0],
            [ 0,-1, 0]]
        )
        self.t = np.array([0.0, 0.0, 1.5])  # camera height in world frame

        # Bird’s-eye ROI & scale → it is arbitrary
        self.x_min, self.x_max = 0.0, 20.0
        self.y_min, self.y_max = -10.0, 10.0
        self.px_per_m = 25 # pixels per meter

        # we need map1 and map2 if we get an image that in not rectified:

        # self.map1 = None         
        # self.map2 = None

        # will be computed at runtime during the first callback
        self.W = None # warp matrix
        self.W_be = None # bird's-eye width
        self.H_be = None # bird's-eye height

        self.bridge = CvBridge()

        # subs and pubs
        self.sub = self.create_subscription(
            Image, '/zed2/rgb/rect', self.cb_image, 10)
        self.pub = self.create_publisher(
            Image, 'image_birdeye', 10)
        
        self.get_logger().info('Birdseye INITIALIZATIO FINISHED')

    def cb_image(self, msg: Image):
        # 1) Convert ros msg to cv2 image
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 2) If we dont have the warp matrix, compute it
        if self.W is None:

            # h, w = img.shape[:2]
            # # undistort maps
            # newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.distCoeffs, (w, h), alpha=0)
            
            # self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.distCoeffs, None, newK, (w, h), cv2.CV_32FC1)

            # compute H_inv
            r1, r2 = self.R[:,0], self.R[:,1]
            M = np.column_stack((r1, r2, self.t))
            H = self.K.dot(M)
            H_inv = np.linalg.inv(H)

            # compute T
            s = self.px_per_m
            x0, y0 = self.x_min, self.y_min
            self.W_be = int((self.x_max - x0) * s)
            self.H_be = int((self.y_max - y0) * s)
            T = np.array([[ s, 0, -s*x0],
                          [ 0, s, -s*y0],
                          [ 0, 0,    1 ]])

            # final warp
            self.W = T.dot(H_inv)

        # # 3) Undistort
        # undist = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

        # 4) Warp to bird’s-eye
        birdseye = cv2.warpPerspective(
            img, self.W, (self.W_be, self.H_be),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
        )

        # 5) Publish
        out_msg = self.bridge.cv2_to_imgmsg(birdseye, 'bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BirdseyeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
