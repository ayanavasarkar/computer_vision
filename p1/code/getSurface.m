function  heightMap = getSurface(surfaceNormals, method)
p = surfaceNormals(:,:,1)./surfaceNormals(:,:,3);
q = surfaceNormals(:,:,2)./surfaceNormals(:,:,3);
[h,w,dim] = size(surfaceNormals);
switch method
    case 'row'
           heightMap = zeros(h,w);
           heightMap(:,1) = q(:,1);
           heightMap(:,1) = cumsum(heightMap(:,1),1);
           heightMap(:,2:w) = p(:,2:w);
           heightMap = cumsum(heightMap,2);
    case 'column'
           heightMap = zeros(h,w);
           heightMap(1,:) = p(1,:);
           heightMap(1,:) = cumsum(heightMap(1,:),2);
           heightMap(2:h,:) = q(2:h,:);
           heightMap = cumsum(heightMap,1);
    case 'average'
        colHeightMap = getSurface(surfaceNormals, 'column');
        rowHeightMap = getSurface(surfaceNormals, 'row');
        heightMap = (colHeightMap+rowHeightMap)./2;
       
    case 'random'
        numberOfIterations = 1000;
        heightMap = zeros(h,w);
        rng(1);
        for i = 1:numberOfIterations
            if(rand()>0.5)
                heightMap = heightMap +...
                    randRowColIntergration(surfaceNormals,p,q);
            else
                heightMap = heightMap +...
                    randColRowIntergration(surfaceNormals,p,q);
            end
        end
        heightMap = heightMap ./ numberOfIterations;
end
end

function heightMapRand = randColRowIntergration(surfaceNormals,p,q)
    [h,w,dim] = size(surfaceNormals);
    heightMapRand = zeros(h, w);
    x = round(rand()*h);
    if(x<2) 
        x = 2;
    end
    for rowIndex = 2:x % 1st column sweep through x rows
        heightMapRand(rowIndex,1) = heightMapRand(rowIndex-1,1) + q(rowIndex,1);
    end

    y = round(rand()*w);
    if(y<2) 
        y = 2;
    end

    for rowIndex = 1:x % sweep 2 to y columns across rows 1 to x in horizontal steps
        for colIndex = 2:y
            heightMapRand(rowIndex, colIndex) = ...
                heightMapRand(rowIndex, colIndex-1)...
                + p(rowIndex,colIndex);
        end
    end
    
    for colIndex = 1:y % sweep h-x to h rows across columns 1 to y in column wise
        for rowIndex = x+1:h
          heightMapRand(rowIndex, colIndex) = ...
              heightMapRand(rowIndex-1, colIndex)...
              + q(rowIndex,colIndex);
        end
    end
    
    for rowIndex = 1:h % complete the remaining area row wise
        for colIndex = y+1:w
            heightMapRand(rowIndex, colIndex) = ...
                heightMapRand(rowIndex, colIndex-1) + p(rowIndex,colIndex);
        end
    end
end

function heightMapRand = randRowColIntergration(surfaceNormals,p,q)
    [h,w,dim] = size(surfaceNormals);
    y = round(rand()*w);
    if(y<2) 
        y = 2;
    end
    heightMapRand = zeros(h, w);
    for colIndex = 2:y % fix firstrow and sweep through y columns
        heightMapRand(1,colIndex) = heightMapRand(1,colIndex-1)...
            + p(1,colIndex);
    end

    x = round(rand()*h);
    if(x<2) 
        x = 2;
    end

    for colIndex = 1:y % sweep through y columns and x rows columnwise
        for rowIndex = 2:x
            heightMapRand(rowIndex, colIndex) = ...
                heightMapRand(rowIndex-1, colIndex)...
                + q(rowIndex,colIndex);
        end
    end

    for colIndex = y+1:w % sweep w-y to w columns across rows 1 to h-x in rowwise
        for rindex = 1:x
            heightMapRand(rowIndex, colIndex) = ...
                heightMapRand(rowIndex, colIndex-1)...
                + p(rowIndex,colIndex);
        end
    end

    for colIndex = 1:w % complete the remaining area column wise
        for rindex = x+1:h
           heightMapRand(rowIndex, colIndex) = ...
               heightMapRand(rowIndex-1, colIndex)...
               + q(rowIndex,colIndex);
        end
    end
end
