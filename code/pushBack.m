% function [f] = pushBack(f, x, y)
% f = [f; x,y];

function [ f, front, back ] = pushBack(f, front, back, x, y)

if (back == front-1) || (front == 1 && back == size(f,1))
    return;
end

f(back,1) = x;
f(back,2) = y;
back = back+1;
if back > size(f,1)
    back = 1;
end