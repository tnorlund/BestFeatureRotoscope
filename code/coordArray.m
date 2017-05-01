function [g, len] = coordArray(img)

if max(max(img)) <= 255 
    g = cell(256,1);
    len = zeros(256,1);
else
    g = cell(max(max(img))+1,1);
    len = zeros(max(max(img))+1,1);
end
%guess the maximum size of memory to allocate
m = (max(max(img))+1)/250;
n = floor(numel(img)/ m);

for ii = 1:numel(g)
    g{ii} = zeros(n,2);
end


for jj = 1:size(img,2)
    for ii = 1:size(img,1)
        v = img(ii,jj)+1;
        len(v) = len(v)+1;
        g{v}(len(v),:) = [ii, jj];
    end
end