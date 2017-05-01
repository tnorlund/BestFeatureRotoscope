function [Gxy, lang] = magnitudeGradient(img, G, Gd)

Gxy = zeros(size(img));
lang = Gxy;

if size(G,2) <= 1
    G = G';
    if size(G,2) <= 1
        return;
    end
end

if size(Gd,2) <= 1
    Gd = Gd';
    if size(Gd,2) <= 1
        return;
    end
end

horizontal = convImg(img,G');
vertical = convImg(img,G);

horizontal = convImg(horizontal,Gd);
vertical = convImg(vertical,Gd');

% h = figure;
% hax = axes('Parent', h);
% showImg(vertical,hax);
% title(hax,'Vertical Image');

% h = figure;
% hax = axes('Parent', h);
% showImg(horizontal,hax);
% title(hax,'Horizontal Image');

for ii = 1:size(img,1)
    for jj = 1:size(img,2)
        Gxy(ii,jj) = sqrt(horizontal(ii,jj)^2 + vertical(ii,jj)^2);
        lang(ii,jj) = atan2(horizontal(ii,jj),vertical(ii,jj));
    end
end