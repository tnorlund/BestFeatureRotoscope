function [Gx, Gy] = getGradient(I, sigma)

[G,w] = gaussian(sigma);
[Gd,wd] = gaussianDerivative(sigma);

Gx = convImg(I,G'); %horiz
Gy = convImg(I,G); %vert

Gx = convImg(Gx,Gd); %horiz
Gy = convImg(Gy,Gd'); %vert


