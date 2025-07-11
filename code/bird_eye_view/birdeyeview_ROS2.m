
node = ros2node("matlab_birdeye");

% subscribe to the image and camera_info topics
imageSub   = ros2subscriber(node,'/zed2/zed_node/rgb/image_rect_color',   'sensor_msgs/Image');
camInfoSub = ros2subscriber(node,'/zed2/zed_node/camera_info',           'sensor_msgs/CameraInfo');

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

figure('Name','ZED2i Live BEV','NumberTitle','off');
while ishandle(gcf)  % run until you close the figure
    % receive the next image (timeout after 10s)
    imgMsg = receive(imageSub, 10);
    if isempty(imgMsg)
        warning('No image received this cycle.');
        continue;
    end

    % convert to MATLAB image
    I = readImage(imgMsg);

    % do your BEV transform
    BEV = transformImage(birdsEye, I);

    % display side‑by‑side
    subplot(1,2,1);
    imshow(I);
    title('Original ZED Image');

    subplot(1,2,2);
    imshow(BEV);
    title('Bird''s-Eye View');

    drawnow;
end

clear imageSub camInfoSub
