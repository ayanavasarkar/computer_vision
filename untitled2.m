gaussianKernel =  ones(3,3);
result_gaussian_kernel = gaussianKernel;
for i=1:10
    result_gaussian_kernel = imfilter(gaussian_kernel,result_gaussian_kernel,'conv');

    plot(result_gaussian_kernel);
end