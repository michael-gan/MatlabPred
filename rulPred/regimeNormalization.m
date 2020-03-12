function data = regimeNormalization(data, centers, centerstats)
% �������ݴ���Ĺ������й�һ������
conditionIdx = 3:5;
dataIdx = 6:26;

% ��ÿһ�в���
data{:, dataIdx} = table2array(...
    rowfun(@(row) localNormalize(row, conditionIdx, dataIdx, centers, centerstats), ...
    data, 'SeparateInputs', false));
end

function rowNormalized = localNormalize(row, conditionIdx, dataIdx, centers, centerstats)
% ��ÿһ�й�һ��

% ��ù����ʹ���������
ops = row(1, conditionIdx);
sensor = row(1, dataIdx);

% �ҳ�������������Ĵ�����
dist = sum((centers - ops).^2, 2);
[~, idx] = min(dist);

% �����������ݸ��������ľ�ֵ�ͱ�׼���һ��
% ��NaN��Infֵ���趨Ϊ0
rowNormalized = (sensor - centerstats.Mean{idx, :}) ./ centerstats.SD{idx, :};
rowNormalized(isnan(rowNormalized) | isinf(rowNormalized)) = 0;
end