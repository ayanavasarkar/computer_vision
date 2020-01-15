% This code is part of:
%
%   CMPSCI 670: Computer Vision
%   University of Massachusetts, Amherst
%   Instructor: Subhransu Maji

%Read test images
img1 = imread('../data/disparity/poster_im2.jpg');
img2 = imread('../data/disparity/poster_im6.jpg');

% Another pair of images
img1 = imread('../data/disparity/tsukuba_im1.jpg');
img2 = imread('../data/disparity/tsukuba_im5.jpg');
img1 = im2double(rgb2gray(img1));
img2 = im2double(rgb2gray(img2));

disparityMap = disparitySGM(img1, img2);
figure;
imshow(disparityMap, [0, 64]);
title('Disparity Map');
colormap jet
colorbar


%Compute depth
depth = depthFromStereo(img1, img2, 23);

Show result
figure(1);
subplot(1,2,1);
imshow(img1);
title('Input image');
subplot(1,2,2);
imshow(img2);
title('Estimated depth map');
