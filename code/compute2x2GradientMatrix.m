function [Z] = compute2x2GradientMatrix(Gx,Gy,featurepoint,window)

ixx = 0;
iyy = 0;
ixy = 0;
i = featurepoint(1);
j = featurepoint(2);
wd = floor(window/2);
[h,w] = size(Gx);

for r = -wd:wd
    for c = -wd:wd
        if(i+r >= 1 && i+r <= h && j+c >= 1 && j+c <= w)
            Gx_interp = Interpolate(Gx,i+r,j+c,h,w);
            Gy_interp = Interpolate(Gy,i+r,j+c,h,w);
            ixx = ixx + Gx_interp^2;
            iyy = iyy + Gy_interp^2;
            ixy = ixy + Gx_interp*Gy_interp;
        end
    end
end
Z = [ixx ixy; ixy iyy];