function [out, numComp, label_img] = connectedCompFF(img)

if size(img,3) == 1
    out = zeros(size(img,1),size(img,2),3);
    on = 255;
else
    out = zeros(size(img));
    on = [255,255,255];
end

label_img = zeros(size(img,1),size(img,2),1);

numComp = 0;
colors = [170 170 170;...
          255 0 0;...
          0 255 0;...
          0 0 255;...
          255 255 0;...
          255 0 255;...
          0 255 255;...
          255 150 150;...
          150 255 150;...
          150 150 255];
[Imax,Jmax,Kmax] = size(img);
for jj = 1:Jmax
    for ii = 1:Imax
        if all(out(ii,jj,:) == 0) && all(img(ii,jj,:) == on)
            numComp = numComp + 1;
            seed = [ii,jj];
            out  = floodfillO(img,seed,cat(3,colors(mod(numComp,10)+1,1),colors(mod(numComp,10)+1,2),colors(mod(numComp,10)+1,3)),out);
            label_img  = floodfillO(img,seed,numComp,label_img);
        end
    end
end