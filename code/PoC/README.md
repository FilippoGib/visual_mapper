# CAMERA CONES DETECTOR
## The aim of this project is to take the bounding boxes of the cones from yolo and return the position of the cones in the car frame. For demonstration purposes we ask the user to manually pick a point at the base of the cones. After the selection the cones will be projected in the 3D world.

### Explaination:
 - We use the intrinsic parameters of the camera:  
    - K:
        |      |      |       |
        |------|------|-------|
        | fx   | s    | cx    |
        | 0    | fy   | cy    |
        | 0    | 0    | 1     |
    - Distortion coefficients:  
        |  |  |  |  |  |
        |--|--|--|--|--|
        | { k1|k2 |k3 |p1 |p2 }
    - Camera extrinsics w.r.t. the car frame: 3D rotation matrix and translation vector

- We get the center of the base of each bounding box $[u,v]$
- Undistort using a cv2 function: $[u,v] -> [x_n, y_n]$
- Create a camera ray going from the camera center to the image plane where $x_n$ and $y_n$ are the coordinates of the cone in the camera after normalization  
    -   $p_{cam}$ = ${(x_n, y_n, 1)}$
- Express the ground plane in the reference frame of the car:
    - versor normal to the plane: $n_{car} = (0,0,1)$
    - height of the plane in d_car: $d_{car} = 0$  

        $n_{car}^T * (X,Y,Z) + d_{car} = 0$
    
- Express the ground plane in the reference frame of the camera:
    - $R^T$ rotation matrix of the camera in the car frame
    - $t$ translation vector of the camer in the car frame
    - $n_{cam} = R^T* n_{car}$ 
    - $d_{cam} = n_{car}^T * t$  

        $n_{cam}^T * (x,y,z) + d_{cam} = 0$

- Intersect the ray and the plane to find the 3D position in the camera frame:  
    - substitute $(x,y,z)$ with the parametrized ray $lambda * p_{cam}$
    - $n_{cam}^T * (lambda * p_{cam}) + d_{cam} = 0$
    - explicit for lambda:  
            
        $lambda = -d_{cam} / (n^T_{cam} * p_{cam})$
    - the point in the camera world is $X_{cam} = lambda * p_{cam}$
- Convert from camera frame to car frame:
    - $X_{car} = X_{cam} * $

    


