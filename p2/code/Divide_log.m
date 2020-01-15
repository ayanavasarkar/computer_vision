function mosim = Divide(im)
	mosim = repmat(im, [1 1 3]); 
	[r, g, b] = getRGBValues(im);
	mosim(:,:,2) = g + imfilter(g, [0 1 0; 1 0 1; 0 1 0]/4);
    g = mosim(:,:,2);
    g(~g) = 0.01;

    r = r ./ g;
    b = b ./ g;

	blueRed = imfilter(b,[1 0 1; 0 0 0; 1 0 1]/4);
	blueGreen = imfilter(b+blueRed,[0 1 0; 1 0 1; 0 1 0]/4);
	b = b + blueRed + blueGreen;


	redBlue = imfilter(r,[1 0 1; 0 0 0; 1 0 1]/4);
	redGreen = imfilter(r+redBlue,[0 1 0; 1 0 1; 0 1 0]/4);
	r = r + redBlue + redGreen;
    mosim(:,:,1) = r .* g; 
    mosim(:,:,3) = b .* g;

function mosim = Log(im)
	mosim = repmat(im, [1 1 3]);
	[r, g, b] = getRGBValues(im);
	
    mosim(:,:,2) = g + imfilter(g, [0 1 0; 1 0 1; 0 1 0]/4);
    g = mosim(:,:,2);

    g(~g) = 0.01; 

    r = r ./ g;
    r(~r) = 0.1; 
    b = b ./ g;
    b(~b) = 0.1;
    r = log(r);
    b = log(b);

	blueRed = imfilter(b,[1 0 1; 0 0 0; 1 0 1]/4);
	blueGreen = imfilter(b+blueRed,[0 1 0; 1 0 1; 0 1 0]/4);
	b = b + blueRed + blueGreen;


    redBlue = imfilter(r,[1 0 1; 0 0 0; 1 0 1]/4);
	redGreen = imfilter(r+redBlue,[0 1 0; 1 0 1; 0 1 0]/4);
	r = r + redBlue + redGreen;
    mosim(:,:,1) = exp(r) .* g;
    mosim(:,:,3) = exp(b) .* g;

function [r, g, b] = getRGBValues(im)
	[h, w] = size(im);
	r = zeros(h, w);
	r(1:2:h, 1:2:w) = im(1:2:h, 1:2:w);

	b = zeros(h, w);
	b(2:2:h, 2:2:w) = im(2:2:h, 2:2:w);

	g = im;
	g(2:2:h, 2:2:w) = 0;
	g(1:2:h, 1:2:w) = 0;