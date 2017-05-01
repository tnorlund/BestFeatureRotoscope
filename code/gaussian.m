function [G, w] = gaussian(sigma)

G = [];
w = [];

if sigma < 0
    return
end

a = round((5 * sigma - 1) / 2);
w = 2 * a + 1;
G = zeros(1,w);
for i = 1:w
    G(i) = exp((-1*(i-1-a)*(i-1-a))/(2*sigma*sigma));
end

G = G/sum(G);