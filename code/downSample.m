function [out] = downSample(img, factor)

[G,w] = gaussian(0.6);

for n = 1:floor(factor/2)
    downSamp = zeros(floor([size(img,1)/2, size(img,2)/2, size(img,3)]));
    img = convImg(img,G);
    img = convImg(img,G');
    [h,w, d] = size(downSamp);
    for k = 1:d
        for ii = 1:h
            for jj = 1:w
                downSamp(ii,jj,k) = img(ii*2,jj*2,k);
            end
        end
    end
    img = downSamp;
end
out = downSamp;