function blobs = detectBlobs(im, par)

if size(im, 3) > 1
    im = rgb2gray(im);
end
if ~isfloat(im)
    im = im2double(im);
end
if nargin < 2
    par.sigma = 2;
    par.interval = 12;
    par.threshold = 1e-2;
    par.display = false;
end
sigma = par.sigma;
r = ceil(3*sigma);
g = fspecial('log', 2*r+1, sigma);
p = scaleSpace(im, par.interval, size(g,1));
p.scores = zeros(size(im,1), size(im,2), length(p.scale));
for i = 1:length(p.scale),
    score = abs(conv2(p.im{i}, g, 'valid'));
    score = padarray(score,[r r], 0, 'both');
    p.score(:,:,i) = imresize(score, size(im));
    if par.display,
        figure(1); clf;
        subplot(1,2,1);
        imagesc(p.im{i}); axis image; colormap gray;
        subplot(1,2,2);
        imagesc(p.score(:,:,i)); axis image; colormap gray;
        pause(0.01);
    end
end
nbhd = ones(3);
for i = 1:length(p.scale),
    ordscore = ordfilt2(p.score(:,:,i), numel(nbhd), nbhd);
    ismax = abs(ordscore - p.score(:,:,i)) < eps;
    p.score(:,:,i) = ismax.*p.score(:,:,i);
    if par.display,
        figure(1); clf;
        subplot(1,2,1);
        imagesc(p.im{i}); axis image; colormap gray;
        subplot(1,2,2);
        imagesc(p.score(:,:,i)); axis image; colormap gray;
        pause(0.01);
    end
end
[blobScore, blobScale] = max(p.score, [], 3);
nmsRadius = max(1,ceil(0.005*sqrt(size(im,1)*size(im,1) + size(im,2)*size(im,2))));
nbhd = ones(2*nmsRadius+1);
ordscore = ordfilt2(blobScore, numel(nbhd), nbhd);
ismax = abs(ordscore - blobScore) < eps;
blobScore = blobScore.*ismax;
inds = blobScore > par.threshold;
[y,x] = find(inds);
blobs = [x y (1./p.scale(blobScale(inds)))'*sigma*sqrt(2) blobScore(inds)];
fprintf('%i points detected (max score=%f)\n', size(blobs,1), max(blobs(:,4)));
function p = scaleSpace(im, interval, minSize)
p.size = size(im);
im = imresize(im, 2);
smallDim = min(size(im, 1), size(im,2));
numOctaves = ceil(log(smallDim/minSize))+1;
p.im = cell(1, numOctaves*interval);
p.scale = zeros(1, numOctaves*interval);
stepSize = 2^(1/interval);
currScale = 2;
offset = 0;
for s = 1:numOctaves, 
    for i = 1:interval,
        p.im{offset+i} = imresize(im, 1/stepSize^(i-1));
        p.scale(offset+i) = currScale/stepSize^(i-1);
    end
    im = imresize(im, 0.5);
    currScale = currScale*0.5;
    offset = offset + interval;
end