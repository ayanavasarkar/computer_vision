function output = demosaicImage(im, method)
switch lower(method)
    case 'baseline'
        output = demosaicBaseline(im);
    case 'nn'
        output = demosaicNN(im);         % Implement this
    case 'linear'
        output = demosaicLinear(im);     % Implement this
    case 'adagrad'
        output = demosaicAdagrad(im);    % Implement this
    case 'divide'
    	output = demosaicTransformDivide(im);
    case 'log'
        output = demosaicTransformLog(im);
end

%--------------------------------------------------------------------------
%                          Baseline demosacing algorithm. 
%                          The algorithm replaces missing values with the
%                          mean of each color channel.
%--------------------------------------------------------------------------
function mosim = demosaicBaseline(im)
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
[h, w] = size(im);

% Red channel (odd rows and columns);
r = im(1:2:h, 1:2:w);
meanValue = mean(mean(r));
mosim(:,:,1) = meanValue;
mosim(1:2:h, 1:2:w,1) = im(1:2:h, 1:2:w);

% Blue channel (even rows and colums);
b = im(2:2:h, 2:2:w);
meanValue = mean(mean(b));
mosim(:,:,3) = meanValue;
mosim(2:2:h, 2:2:w,3) = im(2:2:h, 2:2:w);

% Green channel (remaining places)
% We will first create a mask for the green pixels (+1 green, -1 not green)
mask = ones(h, w);
mask(1:2:h, 1:2:w) = -1;
mask(2:2:h, 2:2:w) = -1;
g = mosim(mask > 0);
meanValue = mean(g);
% For the green pixels we copy the value
greenChannel = im;
greenChannel(mask < 0) = meanValue;
mosim(:,:,2) = greenChannel;

%--------------------------------------------------------------------------
%                           Nearest neighbour algorithm
%--------------------------------------------------------------------------
function mosim = demosaicNN(im)
  mosim = repmat(im, [1 1 3]);
  [h, w] = size(im);
  
  %r channel
  for row = 1:h
      for col =1:w
          if(rem(row,2)==1 && rem(col,2) == 1)
              continue;
          elseif(rem(row,2)==0 && rem(col,2)==1)
              mosim(row,col,1) = mosim(row-1,col,1);
          elseif(rem(row,2)==1 && rem(col,2)==0)
              mosim(row,col,1) = mosim(row,col-1,1);
          elseif(rem(row,2)==0 && rem(col,2)==0)
                  mosim(row,col,1) = mosim(row-1,col-1,1);
          end
      end
  end
  
  %g channel
  for row = 1:h
      for col =1:w
          if((rem(row,2)==0 && rem(col,2)==1) || (rem(row,2)==1 && rem(col,2)==0))
              continue;
          elseif(rem(row,2)==1 && rem(col,2)==1)
              if(col<w)
                  mosim(row,col,2) = mosim(row,col+1,2);
              elseif(col==h)
                  if(row<h)
                      mosim(row,col,2) = mosim(row+1,col,2);
                  else
                      mosim(row,col,2) = mosim(row,col-1,2);
                  end
              end
          elseif(rem(row,2)==0 && rem(col,2)==0)
              mosim(row,col,2) = mosim(row,col-1,2);
          end
          
              
      end
  end  
   %b channel
    for row = 1:h
      for col =1:w
          if(rem(row,2)==0 && rem(col,2)==0)
              continue;%blue pixels
          elseif(rem(row,2)==1 && rem(col,2)==1)
              if(row<h && col<w)
                  mosim(row,col,3) = mosim(row+1,col+1,3);
              elseif(row==h && col~=w)
                  mosim(row,col,3) = mosim(row-1,col+1,3);
              elseif(col==w && row~=h)
                  mosim(row,col,3) = mosim(row+1,col-1,3);
              elseif(row==h && col==w)
                  mosim(row,col,3) = mosim(row-1,col-1,3);
              end
          elseif(rem(row,2)==0 && rem(col,2)==1)
              if(col<w)
                  mosim(row,col,3) = mosim(row,col+1,3);
              elseif(col==w)
                  mosim(row,col,3) = mosim(row,col-1,3);
              end
          elseif(rem(row,2)==1 && rem(col,2)==0)
              if(row<h)
                  mosim(row,col,3) = mosim(row+1,col,3);
              elseif(row==h)
                  mosim(row,col,3) = mosim(row-1,col,3);
              end
          end
      end
    end
%--------------------------------------------------------------------------
%                           Linear interpolation
%--------------------------------------------------------------------------
function mosim = demosaicLinear(im)
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
[r, g, b] = getRGBValues(im);

% Calculate blue pixels at the red location
blueRed = imfilter(b,[1 0 1; 0 0 0; 1 0 1]/4);
% Calculate blue at green location
blueGreen = imfilter(b+blueRed,[0 1 0; 1 0 1; 0 1 0]/4);
mosim(:,:,3) = b + blueRed + blueGreen;

%Calculate red pixels at the blue location
redBlue = imfilter(r,[1 0 1; 0 0 0; 1 0 1]/4);
%Calculate red pixels at the green locations   
redGreen = imfilter(r+redBlue,[0 1 0; 1 0 1; 0 1 0]/4);
mosim(:,:,1) = r + redBlue + redGreen;

% filter for green is same for red and blue pixels
mosim(:,:,2) = g + imfilter(g, [0 1 0; 1 0 1; 0 1 0]/4);

%--------------------------------------------------------------------------
%                           Adaptive gradient
%--------------------------------------------------------------------------
function mosim = demosaicAdagrad(im)
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
	[h, w] = size(im);
	[r, g, b] = getRGBValues(im);

	%Calculate the missing blue pixels at the red location
	blueRed = imfilter(b,[1 0 1; 0 0 0; 1 0 1]/4);
	% Calculate blue at green pixels
	blueGreen = imfilter(b+blueRed,[0 1 0; 1 0 1; 0 1 0]/4);
	mosim(:,:,3) = b + blueRed + blueGreen;

	%  calculate the missing red pixels at the blue location
	redBlue = imfilter(r,[1 0 1; 0 0 0; 1 0 1]/4);
	% Calculate the missing red pixels at the green locations   
	redGreen = imfilter(r+redBlue,[0 1 0; 1 0 1; 0 1 0]/4);
	mosim(:,:,1) = r + redBlue + redGreen;
    outputGreen = g;
    
	for row = 1:h
		for column = 1:w
			if(g(row, column) ~= 0)
				continue;
			end
			if(row > 1 && column > 1 && row < h && column < w)
				if( abs(g(row-1, column) - g(row+1, column)) > abs(g(row, column-1) - g(row, column+1)) )
					outputGreen(row, column) = ( g(row, column-1) + g(row, column+1) )/2;
				else
					outputGreen(row, column) = ( g(row-1, column) + g(row+1, column) )/2;
				end
			elseif( row == 1 || row == h)
				if(row == 1)
					rowToBeUsed = row+1;
				else
					rowToBeUsed = row-1;
				end
				if(column == 1)
					outputGreen(row, column) = ( g(rowToBeUsed, column) + g(row, column+1) )/2;
				elseif(column == w)
					outputGreen(row, column) = ( g(rowToBeUsed, column) + g(row, column-1) )/2;
				else
					outputGreen(row, column) =  ( g(rowToBeUsed, column) + g(row, column-1) + g(row, column+1))/3;
				end
			elseif( column == 1 || column == w )
				if(column == 1)
					colToBeUsed = column+1;
				else
					colToBeUsed = column-1;
				end
				outputGreen(row, column) = ( g(row, colToBeUsed) + g(row-1, column) + g(row+1, column))/3;
			end
		end
	end
	mosim(:,:,2) = outputGreen;





function mosim = demosaicTransformDivide(im)
	mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
	[r, g, b] = getRGBValues(im);
	% the filter for green is same for red and blue pixels
	mosim(:,:,2) = g + imfilter(g, [0 1 0; 1 0 1; 0 1 0]/4);
    g = mosim(:,:,2);
    g(~g) = 0.01; %set 0.01 to zero vals to avoid zero division

    r = r ./ g;
    b = b ./ g;

	%Calculate the missing blue pixels at the red location
	blueRed = imfilter(b,[1 0 1; 0 0 0; 1 0 1]/4);
	% Calculate blue at green, by averaging over 4 surrounding blue values
	blueGreen = imfilter(b+blueRed,[0 1 0; 1 0 1; 0 1 0]/4);
	b = b + blueRed + blueGreen;


	%Calculate the missing red pixels at the blue location
	redBlue = imfilter(r,[1 0 1; 0 0 0; 1 0 1]/4);
	% Calculate the missing red pixels at the green locations
	redGreen = imfilter(r+redBlue,[0 1 0; 1 0 1; 0 1 0]/4);
	r = r + redBlue + redGreen;
    mosim(:,:,1) = r .* g; %Transforming back the pixels
    mosim(:,:,3) = b .* g;

function mosim = demosaicTransformLog(im)
	mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
	[r, g, b] = getRGBValues(im);
	% the filter for green is same for red and blue pixels
	mosim(:,:,2) = g + imfilter(g, [0 1 0; 1 0 1; 0 1 0]/4);
    g = mosim(:,:,2);

    g(~g) = 0.01; %set 0.01 to zero vals to avoid zero division

    r = r ./ g;
    r(~r) = 0.1; %set 0.01 to zero vals
    b = b ./ g;
    b(~b) = 0.1;
    r = log(r);
    b = log(b);

	%Calculate the missing blue pixels at the red location
	blueRed = imfilter(b,[1 0 1; 0 0 0; 1 0 1]/4);
	% Calculate blue at green, by averaging over 4 surrounding blue values
	blueGreen = imfilter(b+blueRed,[0 1 0; 1 0 1; 0 1 0]/4);
	b = b + blueRed + blueGreen;


	%Calculate the missing red pixels at the blue location
	redBlue = imfilter(r,[1 0 1; 0 0 0; 1 0 1]/4);
	% Calculate the missing red pixels at the green locations
	redGreen = imfilter(r+redBlue,[0 1 0; 1 0 1; 0 1 0]/4);
	r = r + redBlue + redGreen;
    %Transforming back
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