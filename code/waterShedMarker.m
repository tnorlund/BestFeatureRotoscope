function [label] = waterShedMarker(Gxy, Marker)

%initialize values
[G, len] = coordArray(double(Gxy));
%label = zeros(size(Gxy)) - 1;
globallabel = 0;
% frontier = [];
frontier = zeros(numel(Gxy),2);
frontier_f = 1;
frontier_b = 1;

[out, numComp, label_img] = connectedCompFF(Marker);
clear('out','var')
clear('numComp','var')

label = label_img - 1;
[h,w] = size(label);

%main loop
for g = 1:256
    temp_label = label;
    
    %part a
    coord = G{g};
    len_g = len(g);
    for n = 1:len_g
        ii = coord(n,1);
        jj = coord(n,2);
        [flag,x,y] = anyN8geqZero(temp_label,ii,jj);
        if flag
            label(ii,jj) = label(x,y);
            % frontier = pushBack(frontier,ii,jj);
            [frontier,frontier_f,frontier_b] = pushBack(frontier,frontier_f,frontier_b,ii,jj);
        end
    end
    
    %part b
    while(frontier_f ~= frontier_b)
        % [frontier,ii,jj] = popFront(frontier);
        [frontier,frontier_f,frontier_b,ii,jj] = popFront(frontier,frontier_f,frontier_b);
        for c = -1:1
            for r = -1:1
                if r ~= 0 || c ~= 0
                    if ii+r > 0 && ii+r <= h && jj+c > 0 && jj+c <= w
                        if Gxy(ii+r,jj+c) <= g-1 && label(ii+r,jj+c) == -1
                            label(ii+r,jj+c) = label(ii,jj);
                            % frontier = pushBack(frontier,ii+r,jj+c);
                            [frontier,frontier_f,frontier_b] = pushBack(frontier,frontier_f,frontier_b,ii+r,jj+c);
                        end
                    end
                end
            end
        end
    end
    
    %part c
    %     for n = 1:size(coord,1)
    %         ii = coord(n,1);
    %         jj = coord(n,2);
    %             if label(ii,jj) == -1
    %                 globallabel = globallabel + 1;
    %                 label = floodfillO(Gxy, [ii,jj], globallabel, label);
    %             end
    %     end
    
end