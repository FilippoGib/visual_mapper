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

% Create ROS2 publisher for the bird's eye view image
bevImagePub = ros2publisher(node, '/bird_eye_view_image', 'sensor_msgs/Image', 1);


imageSub = ros2subscriber(node, ...
    '/zed/zed_node/rgb/image_rect_color',...
    'sensor_msgs/Image',...
    @(~,msg) imageCallback(msg, birdsEye, bevImagePub));

disp('Listening for images. Close the figure window to stop.');

waitfor(imageSub) % Wait for the subscriber to be active. Necessary for clean exit.

% Clean up
clear imageSub camInfoSub bevImagePub

function imageCallback(msg, birdsEye, bevImagePub)
    % Convert ROS image → MATLAB image
    I = readImage(msg);

    % Apply bird’s‑eye transform
    BEV = transformImage(birdsEye, I);

    % Convert the bird's-eye view image back into a ROS message
    bevImageMsg = ros2msg(bevImageMsgType(BEV), BEV);

    % Publish the bird's-eye view image
    send(bevImagePub, bevImageMsg);

    % Clean up explicitly
    clear I BEV bevImageMsg
end
