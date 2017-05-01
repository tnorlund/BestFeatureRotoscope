function [coord] = cornerDetection(img, numCorners, window, sigma)

coord = zeros(numCorners,2);
[Gx, Gy] = getGradient(img, sigma);

[h,w] = size(img);

eigVals = zeros(size(Gx));
for ii = 1:h
    for jj = 1:w
        Z = compute2x2GradientMatrix(Gx,Gy,[ii,jj],window);
        v = eig(Z);
        eigVals(ii,jj) = min(abs(v));
    end
end

n = 0;
while (n < numCorners)
    maxEig = 0;
    index = [0 0];

    [M,r] = max(eigVals);

    [M,c] = max(M);

    index = [r(c), c];

    n = n+1;
    coord(n,:) = index;
    %remove all Eig Values within 4 city blocks of max
    for ii = -2:2
        for jj = -2:2
            if (abs(ii)+abs(jj) <= 2) && (index(1)+ii > 0) && (index(1)+ii <= h) && (index(2)+jj > 0) && (index(2)+jj <= w)
                eigVals(index(1)+ii,index(2)+jj) = 0;
            end
        end
    end
end
