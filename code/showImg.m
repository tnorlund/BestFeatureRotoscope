function [varargout] = showImg(img,varargin)

varargout = {};

if nargin > 1
    h = varargin{1};
    if ischar(h)
        if strcmpi(h,'yes')
            img = uint8(img);
        end
        if nargin == 3
            h = varargin{2};
            if ~ishandle(h)
                h = figure;
                hax = axes('Parent',h);
            end
        end
    else
        img = uint8(img);
    end
    if ~ishandle(h)
        h = figure;
        hax = axes('Parent',h);
    else
        hax = h;
    end
else
    img = uint8(img);
    h = figure;
    hax = axes('Parent',h);
end

imshow(img,'Parent',hax)

switch(nargout)
    case {1}
        varargout{1} = img;
    case {2}
        varargout{1} = h;
        varargout{2} = hax;
    case {3}
        varargout{1} = img;
        varargout{2} = h;
        varargout{3} = hax;
end