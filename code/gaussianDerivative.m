function [Gd, w] = gaussianDerivative(sigma)

Gd = [];
w = [];

if sigma < 0
    return
end

sumGd = 0;
a = round((5 * sigma - 1) / 2);
w = 2 * a + 1;
Gd = zeros(1,w);
for i = 1:w
    Gd(i) = -1*(i-a-1)*exp((-1*(i-1-a)*(i-1-a))/(2*sigma*sigma));
    sumGd = sumGd -i*Gd(i);
end

Gd = Gd/sumGd;