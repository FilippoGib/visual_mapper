% From image create eyebird view given camera intrinsics and estrinsics

focalLength = [530.487 530.603];
principalPoint = [648.425 365.797];
imageSize = [720 1280];

camIntrinsics = cameraIntrinsics(focalLength,principalPoint,imageSize);

height = 1.01;
pitch = 18.0; % degrees downwards

sensor = monoCamera(camIntrinsics,height,'Pitch',pitch);

distAhead = 20;
spaceToOneSide = 6;
bottomOffset = 1;

outView = [bottomOffset,distAhead,-spaceToOneSide,spaceToOneSide];
outImageSize = [NaN,2000];

birdsEye = birdsEyeView(sensor,outView,outImageSize);

I = imread('/home/filippo/Desktop/Università/primo_anno/secondo_semestre/CV/visual_mapper/code/bird_eye_view/pictures/image2.png');

% visualizaition
figure
imshow(I)
title('Original Image')

BEV = transformImage(birdsEye,I);

%imagePoint = vehicleToImage(birdsEye,[20 0]);
%annotatedBEV = insertMarker(BEV,imagePoint);
%annotatedBEV = insertText(annotatedBEV,imagePoint + 5,'20 meters');

figure
imshow(BEV)
title('Bird''s-Eye-View Image: vehicleToImage')

%imagePoint2 = [120 400];
%annotatedBEV = insertMarker(BEV,imagePoint2);

%vehiclePoint = imageToVehicle(birdsEye,imagePoint2);
%xAhead = vehiclePoint(1);
%displayText = sprintf('%.2f meters',xAhead);
%annotatedBEV = insertText(annotatedBEV,imagePoint2 + 5,displayText);

%figure
%imshow(annotatedBEV)
%title('Bird''s-Eye-View Image: imageToVehicle')