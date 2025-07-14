global i;

node = ros2node("matlab_birdeye_publisher");

% create sub for camer info
camInfoSub = ros2subscriber(node, ...
'/zed/zed_node/camera_info', ...
'sensor_msgs/CameraInfo');

% grab one CameraInfo message to build intrinsics
disp('Waiting for camera_info…');
camInfoMsg = receive(camInfoSub, 10);  % wait up to 10 s
if isempty(camInfoMsg)
    error('No CameraInfo received. Is the topic name correct?');
end

% extract focal length, principal point, image size
K              = reshape(camInfoMsg.K, [3,3])';  % row‐major → MATLAB
focalLength    = [K(1,1), K(2,2)];
principalPoint = [K(1,3), K(2,3)];
imageSize      = [camInfoMsg.Height, camInfoMsg.Width];

% build the MATLAB cameraIntrinsics object
camIntrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

% change this if needed
height      = 1.01;
pitch       = 18.0;   % degrees downwards
sensor      = monoCamera(camIntrinsics, height, 'Pitch', pitch);

distAhead       = 20;
spaceToOneSide  = 6;
bottomOffset    = 1;
outView         = [bottomOffset, distAhead, -spaceToOneSide, spaceToOneSide];
outImageSize    = [NaN, 2000];

birdsEye = birdsEyeView(sensor, outView, outImageSize);

birdEyeImagePub = ros2publisher(node, ...
                            '/zed/zed_node/bird_eye_view/image', ...
                            'sensor_msgs/Image');

i = 0;

imageSub   = ros2subscriber(node, ...
                            '/zed/zed_node/rgb/image_rect_color',...
                            'sensor_msgs/Image',...
                            @(~,msg) imageCallback(msg, birdsEye, birdEyeImagePub));

waitfor(node)

% Clean up
clear imageSub camInfoSub bevImagePub


function imageCallback(msg, birdsEye, birdEyeImagePub)
<<<<<<< HEAD
=======
    % Convert ROS image → MATLAB image
    I = readImage(msg);
>>>>>>> 96ff8987abec450ede551bdfa04c47cc0afc62a3

    global i;
    
    if mod(i, 10) == 0
        % Convert ROS image → MATLAB image
        I = readImage(msg);

<<<<<<< HEAD
        % Apply bird’s‑eye transform
        BEV = transformImage(birdsEye, I);
=======
    % Convert BEV to ROS 'sensor_msgs/Image'
    birdEyeImageMsg = ... 
        ros2message('sensor_msgs/Image', ...
                    'Data', BEV, ...
                    'Height', size(BEV, 1), ...
                    'Width', size(BEV, 2), ...
                    'Encoding', 'rgb8', ...
                    'Step', size(BEV, 2) * 3);
>>>>>>> 96ff8987abec450ede551bdfa04c47cc0afc62a3

        % Convert BEV to ROS 'sensor_msgs/Image'
        birdEyeImageMsg = ... 
            ros2message('sensor_msgs/Image', ...
                        'Data', BEV, ...
                        'Height', size(BEV, 1), ...
                        'Width', size(BEV, 2), ...
                        'Encoding', 'rgb8', ...
                        'Step', size(BEV, 2) * 3);

        % Publish the bird’s-eye view image
        send(birdEyeImagePub, birdEyeImageMsg);

        clear BEV
        i = 0;
    end
    i = i + 1;
end
