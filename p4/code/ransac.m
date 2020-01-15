function [inliers, transf] = ransac(matches, c1, c2)
if nargin < 4
    method = 'affine';
end
c1 = c1';   c2 = c2';
maxIter = 500;
valid = find(matches > 0);
src = c1(1:2, valid);
dst = c2(1:2, matches(valid));
numValid = length(valid);
maxInliers = 3;
for iter = 1:maxIter,
    sel = randperm(numValid);
    switch method
        case 'affine'
            sel = sel(1:3);
            src1 = src(:, sel);
            dst1 = dst(:, sel);
            Arr = estimateAffine(src1, dst1);
            tdst = Arr*[dst; ones(1, size(dst, 2))];
            fn = @estimateAffine;
        otherwise
            error('Wrong Option');
    end
    err = sum((tdst - src).^2);
    isInlier = err < 10;
    currInliers = sum(isInlier);
    if currInliers > maxInliers
        maxInliers = currInliers;
        transf = fn(src(:,isInlier), dst(:,isInlier));
        inliers = valid(isInlier);
    end
end
fprintf(['RANSAC finished with %i iters [%i inliers]\n' ...
                'affine transformation = [%.2f, %.2f, %.2f; %.2f, %.2f, %.2f]\n'], ...
                iter, maxInliers,transf(1),transf(3),transf(5),transf(2),transf(4),transf(6));
function transf = estimateAffine(src, dst)
assert(size(src, 2) == size(dst, 2), 'different size of source and target')
assert(size(dst, 2) >= 3, 'minimum number of 4 matches required for estimating affine transformation')
Arr = kron(eye(2), dst');
Arr = cat(2, A, kron(eye(2), ones(size(dst, 2), 1)));
b = [src(1,:), src(2,:)]';
M = Arr\b;
transf = [M(1), M(2), M(5); M(3), M(4), M(6)];
function transf = estimateHomography(src, dst)
assert(size(src, 2) == size(dst, 2), 'different size of source and target')
assert(size(dst, 2) >= 4, 'minimum number of 4 matches required for estimating affine transformation')
Arr = kron(eye(2), -dst');
Arr = cat(2, Arr, kron(eye(2), -ones(size(dst, 2), 1)));
a1 = [src(1,:).*dst(1,:) src(2,:).*dst(1,:)]';
a2 = [src(1,:).*dst(2,:) src(2,:).*dst(2,:)]';
a3 = [src(1,:) src(2,:)]';
Arr = [Arr, a1, a2 ,a3];
cols = [1, 2, 5, 3, 4, 6, 7, 8, 9];
Arr = Arr(:, cols);
[~, ~, V] = svd(Arr);
transf = reshape(V(:,end), [3, 3]);
transf = transf';