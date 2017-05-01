function out = floodfillO(img, seed, newColor, varargin)
% This function is the same as floodfill, but it can use both the same
% image for the output or a different image for the output based on the
% input parameters.
% out = floodfillO(img, seed, newColor) 
%   this flood fills img and returns it
% out = floodfillO(refImg, seed, newColor, outImg)
%   this flood fills outImg using refImg and returns a modified outImg.

if nargin > 3
    out = varargin{1};
else
    out = img;
end
if size(out,3) == 3
    if numel(newColor) == 1
        newColor = cat(3,newColor,newColor,newColor);
    end
elseif numel(newColor) == 3
    newColor = mean(newColor); 
end
frontier = zeros(numel(img),2);
frontCount = 1;
frontier(frontCount,:) = seed;
oldColor = img(seed(1),seed(2),:);
out(seed(1),seed(2),:) = newColor;
[h, w, nc] = size(img);
clear('nc');
while ~isempty(frontier) && frontCount ~= 0
    %pop the last entry out of frontier
    currPix = frontier(frontCount,:);
    frontCount = frontCount - 1;
    %push all valid neighbors onto frontier
    if currPix(1) > 1
        if all(img(currPix(1)-1,currPix(2),:) == oldColor) && any(out(currPix(1)-1,currPix(2),:) ~= newColor)
            out(currPix(1)-1,currPix(2),:) = newColor;
            %push
            frontCount = frontCount + 1;
            frontier(frontCount,:) = [currPix(1)-1,currPix(2)];
            
        end
    end
    if currPix(1) < h
        if all(img(currPix(1)+1,currPix(2),:) == oldColor) && any(out(currPix(1)+1,currPix(2),:) ~= newColor)
            out(currPix(1)+1,currPix(2),:) = newColor;
            %push
            frontCount = frontCount + 1;
            frontier(frontCount,:) = [currPix(1)+1,currPix(2)];
            
        end
    end
    if currPix(2) > 1
        if all(img(currPix(1),currPix(2)-1,:) == oldColor) && any(out(currPix(1),currPix(2)-1,:) ~= newColor)
            out(currPix(1),currPix(2)-1,:) = newColor;
            %push
            frontCount = frontCount + 1;
            frontier(frontCount,:) = [currPix(1),currPix(2)-1];
        end
    end
    if currPix(2) < w
        if all(img(currPix(1),currPix(2)+1,:) == oldColor) && any(out(currPix(1),currPix(2)+1,:) ~= newColor)
            out(currPix(1),currPix(2)+1,:) = newColor;
            %push
            frontCount = frontCount + 1;
            frontier(frontCount,:) = [currPix(1),currPix(2)+1];
        end
    end
end
