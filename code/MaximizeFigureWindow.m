% Function to maximize the window via undocumented Java call.
% Reference: http://undocumentedmatlab.com/blog/minimize-maximize-figure-window
function MaximizeFigureWindow(varargin)

if nargin == 1
    if ishandle(varargin{1})
        h = varargin{1};
    else
        h = handle(gcf);
    end
    
else
    h = handle(gcf);
end

try
    FigurejFrame = get(h,'JavaFrame');
    FigurejFrame.setMaximized(true);
catch ME
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
        ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage);
    uiwait(warndlg(errorMessage));
end

return; % from MaximizeFigureWindow()