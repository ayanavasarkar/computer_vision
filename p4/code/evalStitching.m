% Evaluation code for image stitching. 
% Your goal is to implement image stitching with RANSAC 
%
% This code is part of:
%
%   CMPSCI 670: Computer Vision
%   University of Massachusetts, Amherst
%   Instructor: Subhransu Maji

%% Set up the library for extracting SIFT
osinfo = computer;
switch osinfo
    case 'MACI64'
        addpath('mex/mexmaci64/');
    case 'GLNXA64'
        addpath('mex/mexa64/');
    case 'PCWIN64'
        addpath('mex/mexw64/');
    case 'PCWIN'
        addpath('mex/mexw32/');
    case 'GLNX86'
        addpath('mex/mexglx/');
end


%% image directory
dataDir = fullfile('..','data', 'stitching');

%% Read input images
testExamples = {'hill', 'field', 'ledge', 'pier', 'river' 'roofs', 'building', 'uttower'};
exampleIndex = 6;
imageName1 = sprintf('%s1_r.jpg', testExamples{exampleIndex});
imageName2 = sprintf('%s2_r.jpg', testExamples{exampleIndex});
im1 = imread(fullfile(dataDir, imageName1));
im2 = imread(fullfile(dataDir, imageName2));

%% Detect keypoints
blobs1 = detectBlobs(im1);  % implement this for evalBlobsDetection
blobs2 = detectBlobs(im2);

%% Compute SIFT features
sift1 = compute_sift(im1, blobs1(:, 1:3));
sift2 = compute_sift(im2, blobs2(:, 1:3));

%% Find the matching between features
matches = computeMatches(sift1, sift2);  % implement this
showMatches(im1, im2, blobs1, blobs2, matches);

%% Ransac to find correct matches and compute transformation
[inliers, transf] = ransac(matches, blobs1, blobs2);  % implement this

goodMatches = zeros(size(matches));
goodMatches(inliers) = matches(inliers);

showMatches(im1, im2, blobs1, blobs2, goodMatches);

%% Merge two images and display the output
stitchIm = mergeImages(im1,im2, transf);
figure;
imshow(stitchIm);
title(sprintf('stitched image: %s', testExamples{exampleIndex}));