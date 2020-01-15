im = imread('lion.png');

x = im;
y = zeros(size(x));
for c_channel = 1:size(x,3)
    for i = 1:size(x,1)
        for j = 1:size(x,2)
            if(j>1)
                y(i,j,c_channel) = x(i,j-1,c_channel);
            else
                if(i>1)
                    y(i,j,c_channel) = y(i-1,j,c_channel);
                else
                    y(i,j,c_channel) = y(i+1,j+1,c_channel);
                end
            end
        end
    end
end
h = ones(3,3)/9;
x = imfilter(x,h);
y = imfilter(y,h);
x = reshape(x,1,[]);
y = reshape(y,1,[]);
subplot(1, 2, 1);
scatter(x, y);
xlabel('Pixel Intensity Value')
ylabel('Left Pixel Intensity Value')


x = im;
y = zeros(size(x));
n = size(x,1);
m = size(x,2);
for c_channel = 1:size(x,3)
    for i = 2:n-1
        for j = 2:m-1
            y(i,j,c_channel)=(x(i,j-1,c_channel) + x(i,j+1,c_channel) + x(i-1,j,c_channel) +x(i+1,j,c_channel))/4;
        end
    end
end
for c_channel = 1:size(x,3)
    for i = 2:n-1
        y(i,1,c_channel) = (x(i-1,1,c_channel) + x(i+1,1,c_channel) + x(i,2,c_channel) + x(i,m,c_channel))/4;
        y(i,m,c_channel) = (x(i-1,m,c_channel) + x(i+1,m,c_channel) + x(i,m-1,c_channel) + x(i,1,c_channel))/4;
    end
    for j = 2:m-1
        y(1,j,c_channel) = (x(1,j-1,c_channel) + x(1,j+1,c_channel) + x(2,j,c_channel) + x(n,j,c_channel))/4;
        y(n,j,c_channel) = (x(n,j-1,c_channel) + x(n,j+1,c_channel) + x(n-1,j,c_channel)+ x(1,j,c_channel))/4;
    end
    y(1,1,c_channel) = (x(1,2,c_channel) + x(2,1,c_channel) + x(1,m,c_channel) + x(n,1,c_channel))/4;
    y(n,1,c_channel) = (x(n-1,1,c_channel) + x(n,2,c_channel) + x(n,m,c_channel) + x(1,1,c_channel))/4;
    y(1,m,c_channel) = (x(1,m-1,c_channel) + x(2,m,c_channel) + x(1,1,c_channel) + x(n,1,c_channel))/4;
    y(n,m,c_channel) = (x(n-1,m,c_channel) + x(n,m-1,c_channel) + x(1,m,c_channel) + x(n,1,c_channel))/4;
end
h = ones(3,3)/9;
x = imfilter(x,h);
y = imfilter(y,h);
len = size(x);
x = reshape(x,1,[]);
y = reshape(y,1,[]);
subplot(1, 2, 2);
scatter(x, y);
xlim([0 100]);

xlabel('Pixel Intensity Value')
ylabel('Neighboring Pixels Average Intensity Value')