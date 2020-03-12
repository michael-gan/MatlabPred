function writeMFPTBearing(filename, data)
% Append data to a file, used by the WriteToMemberFcn of a
% fileEnsembleDatastore.
%
% Inputs:
% filename - string with name of file to write to
% data     - a structure with data to write

% Copyright 2017-2018 The MathWorks, Inc.

save(filename, '-append', '-struct', 'data');
end