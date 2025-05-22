# CAMERA CONES DETECTOR
## The aim of this node is to take the bounding boxes from yolo and return the position of the cones in the car frame  

### Math:
 - You are gonna need the intrinsic parameters of the camera:  
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

- Get the center of the base of each bounding box $[u,v]$
- Undistort using a cv2 function: $[u,v] -> [x_n, y_n]$
- Create a camera ray going from the camera center to the image plane where $x_n$ and $y_n$ are the coordinates of the cone in the camera after normalization  
    -   $p_{cam}$ = {$x_n, y_n, 1$}
- Express the ground plane in the reference frame of the car:
    - versore normale al piano: $n_{car} = (0,0,1)$
    - altezza del piano in d_car: $d_{car} = 0$  

        $n_{car}^T * (X,Y,Z) + d_{car} = 0$
    
- Express the ground plane in the reference frame of the camera:
    - $R^T$ matrice di rotazione della camera nel car frame
    - $t$ vettore di traslazione della camera nel car frame
    - $n_{cam} = R^T* n_{car}$ 
    - $d_{cam} = n_{car}^T * t$  

        $n_{cam}^T * (x,y,z) + d_{cam} = 0$

- Intersect the ray and the plane to find the 3D position in the camera frame:  
    - sostituisco a (x,y,z) il mio raggio parametrizzato $lambda * p_{cam}$
    - $n_{cam}^T * (lambda * p_{cam}) + d_{cam} = 0$
    - explicito per lambda:  
            
        $lambda = -d_{cam} / (n^T_{cam} * p_{cam})$
    - il mio punto nel mondo della camera è $X = lambda * p_{cam}$
    


