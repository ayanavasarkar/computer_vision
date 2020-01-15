function [imShift, predShift] = alignChannels(im, maxShift)

% Sanity check
assert(size(im,3) == 3);
assert(all(maxShift > 0));

% Dummy implementation (replace this with your own)
predShift = zeros(2, 2);
%imShift = im;

step=1;
% edges = edge(im,'Prewitt');
r = edge(im(:,:,1),'approxcanny');
b = edge(im(:,:,2),'approxcanny');
g = edge(im(:,:,3),'approxcanny');

min_ssd_b = inf;
min_ssd_g = inf;

for y = -maxShift:step:maxShift
    for x = -maxShift:step:maxShift

        t1 = circshift(b, [x y]);
        t2 = circshift(g, [x y]);

        ssd_bchannel = sum(sum((r-t1).^2));

        ssd_gchannel = sum(sum((r-t2).^2));

        if ssd_bchannel < min_ssd_b
            min_ssd_b = ssd_bchannel;
            predShift(1,:) = [x y];
        end

        if ssd_gchannel < min_ssd_g
            min_ssd_g = ssd_gchannel;
            predShift(2,:) = [x y];

        end
    end
end

newB = circshift(im(:,:,2), predShift(1,:));
newG = circshift(im(:,:,3), predShift(2,:));
imShift = cat(3, im(:,:,1), newB, newG);