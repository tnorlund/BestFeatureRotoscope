function [ outputFrame ] = rotoscope...
    (numberOfFeatures, backgroundFrame, foregroundFrame )
%rotoscope - Uses Watershed segmentation based off of Eigenvalues
%   Detailed explanation goes here

factor = 4;
foreGround = [165 35 100 100];
foreGroundFeature = 1;
gforegroundFrame = rgb2gray(foregroundFrame);
showImg(gforegroundFrame);

invert = rgb2gray(uint8(abs(double(backgroundFrame) - double(foregroundFrame))));
dsInvert = downSample(invert, factor);

corners = cornerDetection(dsInvert ,numberOfFeatures,6,0.6);
center = floor(mean(corners)*factor);


[h,w] = size(dsInvert);

corner_img = zeros(size(invert));

for ii = 1:h
    for jj = 1:w
        if any(corners(:,1) == ii & corners(:,2) == jj)
            corner_img(floor((ii-0.5)*factor),floor((jj-0.5)*factor)) = 255;
        end
    end
end

[h,w] = size(invert);

if (    center(1)-foreGround(1) >  0 &&...
        center(1)+foreGround(2) <= h &&...
        center(2)-foreGround(3) >  0 &&...
        center(2)+foreGround(4) <= w)
    corners1 = cornerDetection(invert(center(1)-foreGround(1):center(1)+foreGround(2),center(2)-foreGround(3):center(2)+foreGround(4)) ,numberOfFeatures/foreGroundFeature,6,0.6);
    corners1 = corners1 + [ones(size(corners1,1),1)*(center(1)-foreGround(1)) ones(size(corners1,1),1)*(center(2)-foreGround(3))];

    for ii = center(1)-foreGround(1):center(1)+foreGround(2)
        for jj = center(2)-foreGround(3):center(2)+foreGround(4)
            if any(corners1(:,1) == ii & corners1(:,2) == jj)
                corner_img(ii,jj) = 255;
            end
        end
    end
end
[G,~] = gaussian(0.6);
[Gd,~] = gaussianDerivative(0.6);
[Gxy,~] = magnitudeGradient(gforegroundFrame, G, Gd);

[label] = waterShedMarker(uint8(Gxy), corner_img);
if min(min(label)) < 0
    label = label-min(min(label));
end

outputFrame = zeros(size(foregroundFrame));

[g, len] = coordArray(label);

for n = 1:max(max(label))+1
    R = 0;
    G = 0;
    B = 0;
    gg = g{n};
    count = len(n);

    for ll = 1:count
        R = R + double(foregroundFrame(gg(ll,1),gg(ll,2),1));
        G = G + double(foregroundFrame(gg(ll,1),gg(ll,2),2));
        B = B + double(foregroundFrame(gg(ll,1),gg(ll,2),3));
    end
    outputFrame = uint8(outputFrame);
    R = R / count;
    G = G / count;
    B = B / count;
    for ll = 1:count
        outputFrame(gg(ll,1),gg(ll,2),1) = R;
        outputFrame(gg(ll,1),gg(ll,2),2) = G;
        outputFrame(gg(ll,1),gg(ll,2),3) = B;
    end
end

end
