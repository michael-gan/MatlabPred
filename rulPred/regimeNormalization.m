function data = regimeNormalization(data, centers, centerstats)
% 根据数据簇类的归属进行归一化处理
conditionIdx = 3:5;
dataIdx = 6:26;

% 对每一行操作
data{:, dataIdx} = table2array(...
    rowfun(@(row) localNormalize(row, conditionIdx, dataIdx, centers, centerstats), ...
    data, 'SeparateInputs', false));
end

function rowNormalized = localNormalize(row, conditionIdx, dataIdx, centers, centerstats)
% 对每一行归一化

% 获得工况和传感器数据
ops = row(1, conditionIdx);
sensor = row(1, dataIdx);

% 找出离样本点最近的簇中心
dist = sum((centers - ops).^2, 2);
[~, idx] = min(dist);

% 将传感器数据根据其簇类的均值和标准差归一化
% 将NaN和Inf值都设定为0
rowNormalized = (sensor - centerstats.Mean{idx, :}) ./ centerstats.SD{idx, :};
rowNormalized(isnan(rowNormalized) | isinf(rowNormalized)) = 0;
end