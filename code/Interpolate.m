function [Jvalue]=Interpolate(J,inew,jnew,height,width)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes 

i0=floor(inew);
j0=floor(jnew);
i1=round(inew);
j1=round(jnew);
%[height,width,colors]=size(J);

if(i0>0 && j0>0 && i0<=height && j0<=width)
    f00=J(i0,j0);
else
    f00=0;
end
if (i1<=height && i1>0 && j0>0 && j0<=width)
    f01=J(i1,j0);
else
    f01=0;
end
if (i0>0 && i0<=height && j1<=width && j1>0)
    f10=J(uint32(i0),uint32(j1));
else
    f10=0;
end
if (i1<=height && i1>0 && j1<=width && j1>0)
    f11=J(i1,j1);
else
    f11=0;
end

alpha=jnew-j0;
beta=inew-i0;

Jvalue=(1-beta)*(1-alpha)*f00+(1-beta)*alpha*f10+beta*(1-alpha)*f01+beta*alpha*f11;
end

