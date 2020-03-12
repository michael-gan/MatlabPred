function helperPlotCombs(ncomb, f)
%HELPERPLOTCOMBS Plot harmonic cursors on a power spectrum plot

% Copyright 2017 The MathWorks, Inc.
ylimit = get(gca, 'YLim');
ylim(ylimit);
ycomb = repmat([ylimit nan], 1, ncomb);
hold(gca, 'on')
for i = 1:length(f)
    xcomb = f(i)*(1:ncomb);
    xcombs = [xcomb; xcomb; nan(1, ncomb)];
    xcombs = xcombs(:)';
    plot(xcombs, ycomb, '--')
end
hold(gca, 'off')
end