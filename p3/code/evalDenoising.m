% Entry code for evaluating demosaicing algorithms
% The code loops over all images and methods, computes the error and
% displays them in a table.
% 
%
% This code is part of:
%
%   CMPSCI 670: Computer Vision
%   University of Massachusetts, Amherst
%   Instructor: Subhransu Maji
%
% Load images
im = im2double(imread('../data/denoising/saturn.png'));
noise1 = im2double(imread('../data/denoising/saturn-noise1g.png'));
noise2 = im2double(imread('../data/denoising/saturn-noise1sp.png'));
noise3 = im2double(imread('../data/denoising/saturn-noise2g.png'));
noise4 = im2double(imread('../data/denoising/saturn-noise2sp.png'));

error1 = sum(sum((im - noise1).^2));
error2 = sum(sum((im - noise2).^2));
error3 = sum(sum((im - noise3).^2));
error4 = sum(sum((im - noise4).^2));
currNoise = noise1;
% fprintf('Input, Errors: %.2f %.2f %.2f %.2f\n', error1, error2,error3,error4);
% Display the images
figure(1);

%% Denoising algorithm (Gaussian filtering)
min_error = inf;
min_signma = 0;
result = 0;
for i = 0.5:0.1:5
    result_gaussian = imgaussfilt(currNoise,i);
    error = sum(sum((im - result_gaussian).^2));
    if(error < min_error)
        min_error = error;
        min_sigma = i;
        result = result_gaussian;
    end
end

subplot(3,3,1); imshow(im); title('Input');
subplot(3,3,2); imshow(currNoise);title(sprintf('Noisy Image1 error %.2f', error1));
subplot(3,3,3); imshow(result);title(sprintf('Gaussian %.2f', min_error));

fprintf('Gaussian  Errors and sigma: %.2f %.2f\n', min_error, min_sigma);

%% Denoising algorithm (Median filtering)
min_error = inf;
min_i =0;
min_j =0;
result =0;
for i = 1:10
    for j = 1:10
    result_median = medfilt2(currNoise,[i j]);
    error = sum(sum((im - result_median).^2));
    if(error < min_error)
        min_error = error;
        min_i = i;
        min_j = j;
        result = result_median;
    end
    end
end

subplot(3,3,4); imshow(im); title('Input');
subplot(3,3,5); imshow(currNoise); title(sprintf('Noisy Image1 error %.2f', error1));
subplot(3,3,6); imshow(result);title(sprintf('Median filter %.2f', min_error));
fprintf('Median Filter  Errors and [i j]: %.2f %d %d\n', min_error, min_i,min_j);


%% Denoising alogirthm (Non-local means)
min_error = inf;
min_patchRadius = 0.0;
min_windowradius = 0.0;
result =0;
for i=2:3
    for j= 5:9
        dn_im = nlMeans(currNoise, i,j, 1);
        error2 = sum(sum((im - dn_im).^2));
        disp(j);
        if(error2<min_error)
            min_error = error2;
            min_patchRadius = i;
            min_windowradius = j;
            result = dn_im;
        end
    end
    disp(i);
end

fprintf('Non-local means  Errors : %.2f Patch Radius = %.2f Window Radius= %.2f\n', min_error,min_patchRadius,min_windowradius);

subplot(3,3,7); imshow(im); title('Input');
subplot(3,3,8); imshow(currNoise);title(sprintf('Noisy Image1 error %.2f', error1));
subplot(3,3,9); imshow(result);title(sprintf('Non local means %.2f', min_error));
function [output]=nlMeans(im, patchRadius, windowRadius, gamma)
   % Size of the image
   [m,n]=size(im);
   output = im;

  % pad the boundaries of the input image
  im = padarray(im,[patchRadius patchRadius],'symmetric');

  widthMin = 1 + patchRadius; %actual image start row pixels
  widthMax = m + patchRadius; %actual image end row pixels
  heightMin = 1 + patchRadius; %actual image start column pixels
  heightMax = n + patchRadius; %actual image end column pixels

  %Loop through all pixels of image where the current pixel will be center
  for i = widthMin : widthMax
  for j = heightMin : heightMax

    currentPixelPatch =  im(i-patchRadius:i+patchRadius,...
        j-patchRadius:j+patchRadius); %current pixel center

    % Window patch limits fixed
    window_row_begin = max(i-windowRadius, widthMin);
    window_row_end = min(i+windowRadius, widthMax);
    window_col_begin = max(j-windowRadius, heightMin);
    window_col_end = min(j+windowRadius, heightMax);

    sumOfWeights = 0;
    weightedIntensitySum = 0;

    %Looping through pixels in the window
    for wind_row = window_row_begin : window_row_end
    for wind_col = window_col_begin : window_col_end

      tempPatch = im(wind_row-patchRadius:wind_row+...
          patchRadius, wind_col-patchRadius:wind_col+patchRadius);
      %Calculate the P(x) - P(y)
      tempSSD = getSquaredDifference(currentPixelPatch, tempPatch);
      %Calculate the exponential
      tempWeight = exp( -tempSSD/ (gamma*gamma));

      sumOfWeights = sumOfWeights + tempWeight;
      weightedIntensitySum = weightedIntensitySum +...
          tempWeight*im(wind_row, wind_col);
    end
    end

  output(i-patchRadius, j-patchRadius) = weightedIntensitySum / sumOfWeights;
  end
  end
end


function errorx = getSquaredDifference(im1, im2)
  errorx = sum(sum((im1 - im2).^2));
end