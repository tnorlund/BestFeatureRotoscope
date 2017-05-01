function [out, varargout] = convImg(img, se)
% out = convImg(img, se)
%   This function preforms convolution on the image (img) based on the
%   kernal (se). Only 2D images should be input into the function for se.
%   Img can be either color or grayscale. Out is the same size as img.

varargout = {};

if size(se,3) == 3 %input validation
    out = [];
    return;
end

img = double(img);

out = zeros(size(img));

center = floor(size(se)*0.5)+1;
offset = size(se);

se = rot90(se,2); %flip se

if size(img,3) > 1
    for k = 1:size(img,3)
        for ii = 1:size(img,1)
            for jj = 1:size(img,2)
                summ = 0;
                for r = 1:offset(1)
                    for c = 1:offset(2)
                        if ii+r-center(1) > 0 && ii+r-center(1) <= size(img,1) && jj+c-center(2) > 0 && jj+c-center(2) <= size(img,2)
                            summ = summ + img(ii+r-center(1),jj+c-center(2),k)*se(r,c);
                        end
                    end
                end
                out(ii,jj,k) = summ;
            end
        end
    end
else
    for ii = 1:size(img,1)
        for jj = 1:size(img,2)
            summ = 0;
            for r = 1:offset(1)
                for c = 1:offset(2)
                    if ii+r-center(1) > 0 && ii+r-center(1) <= size(img,1) && jj+c-center(2) > 0 && jj+c-center(2) <= size(img,2)
                        summ = summ + img(ii+r-center(1),jj+c-center(2))*se(r,c);
                    end
                end
            end
            out(ii,jj) = summ;
        end
    end
end