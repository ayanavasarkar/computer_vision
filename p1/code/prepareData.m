function output = prepareData(imArray, ambientImage)

output = imArray - ambientImage;
output(output < 0) = 0;

maxVal = max(imArray(:));
output = output ./maxVal;