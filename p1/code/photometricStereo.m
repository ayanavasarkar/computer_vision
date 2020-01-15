function [albedoImage, surfaceNormals] = photometricStereo(imArray, lightDirs)

[height, width, ndim] = size(imArray);
albedoImage = zeros(height,width);
surfaceNormals = zeros(height,width,3);

imageVector = reshape(imArray,height*width,ndim);
gfunc = imageVector / lightDirs.';
gfunc = reshape(gfunc,height,width,3);


albedoImage = sqrt(sum(gfunc.^2, 3));
surfaceNormals = bsxfun(@rdivide, gfunc, albedoImage);
