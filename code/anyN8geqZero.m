function [flag,x,y] = anyN8geqZero(img,ii,jj)

[h,w,n] = size(img);
%clear('n','var');

x = 0;
y = 0;
flag = 0;

for c = -1:1
    for r = -1:1
        if r ~= 0 || c ~= 0
            if ii+r > 0 && ii+r <= h && jj+c > 0 && jj+c <= w
                if img(ii+r,jj+c) >= 0
                    flag = 1;
                    x = ii+r;
                    y = jj+c;
                    break;
                end
            end
        end
    end
    if flag
        break;
    end
end