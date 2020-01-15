function depth = depthFromStereo(img1, img2, ws)
% Implement this
depth = ones(size(img1,1), size(img1,2));
img1 = im2double(rgb2gray(img1));
img2 = im2double(rgb2gray(img2));

disparityMap = disparitySGM(frameLeftGray, frameRightGray);
figure;
imshow(disparityMap, [0, 64]);
title('Disparity Map');
colormap jet
colorbar

[nRows,nCols] = size(depth);

% Pad both images with 0
img1 = padarray(img1,[ws ws],0,'both');
img2 = padarray(img2,[ws ws],0,'both');

disp_mat = zeros(size(img1,1), size(img1,2));

% Calculate the displacement for each pixel
for i = ws+1:nRows+ws
    for j = ws+1:nCols+ws
        minSSD = inf;
        disp_x = 0;
        patch1 = img1(i-ws:i+ws,j-ws:j+ws);
        for k = ws+1:nCols+ws
                patch2 = img2(i-ws:i+ws,k-ws:k+ws);
%               Using SSD to find the correspondences
                ssd = sum(sum((patch1-patch2).^2));
                if ssd < minSSD
                    minSSD = ssd;
                    disp_x = abs(k-j);
                end
        end
    disp_mat(i-ws,j-ws) = disp_x;
    end
end
% Calculate Depth using Disparity since b.f=1
depth = 1./disp_mat;
end