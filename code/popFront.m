% function [f,x,y] = popFront(f)

% x = f(1,1);
% y = f(1,2);
% f(1,:) = [];

function [ f, front, back, x, y ] = popFront(f, front, back)

if front == back
    x = [];
    y = [];
else
    x = f(front,1);
    y = f(front,2);
    front = front+1;
    if front > size(f,1)
        front = 1;
    end
end