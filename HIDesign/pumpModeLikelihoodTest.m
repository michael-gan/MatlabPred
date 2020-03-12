function pumpModeLikelihoodTest(HealthyTheta, LargeTheta, SmallTheta)
%pumpModeLikelihoodTest Generate predictions based on PDF values and plot confusion matrix.
%mvnpdf函数用来计算概率密度函数值

m1 = mean(HealthyTheta);
c1 = cov(HealthyTheta);
m2 = mean(LargeTheta);
c2 = cov(LargeTheta);
m3 = mean(SmallTheta);
c3 = cov(SmallTheta);

N = size(HealthyTheta,1);

% True classes
% 1: Healthy: group label is 1.
X1t = ones(N,1);
% 2: Large gap: group label is 2.
X2t = 2*ones(N,1);
% 3: Small gap: group label is 3.
X3t = 3*ones(N,1);

% Compute predicted classes as those for which the joint PDF has the maximum value.
X1 = zeros(N,3); 
X2 = zeros(N,3); 
X3 = zeros(N,3); 
for ct = 1:N
   % Membership probability density for healthy parameter sample
   HealthySample  = HealthyTheta(ct,:);
   x1 = mvnpdf(HealthySample, m1, c1);
   x2 = mvnpdf(HealthySample, m2, c2);
   x3 = mvnpdf(HealthySample, m3, c3);
   X1(ct,:) = [x1 x2 x3];
   
   % Membership probability density for large gap pump parameter
   LargeSample  = LargeTheta(ct,:);
   x1 = mvnpdf(LargeSample, m1, c1);
   x2 = mvnpdf(LargeSample, m2, c2);
   x3 = mvnpdf(LargeSample, m3, c3);
   X2(ct,:) = [x1 x2 x3];
   
   % Membership probability density for small gap pump parameter
   SmallSample  = SmallTheta(ct,:);
   x1 = mvnpdf(SmallSample, m1, c1);  
   x2 = mvnpdf(SmallSample, m2, c2);
   x3 = mvnpdf(SmallSample, m3, c3);
   X3(ct,:) = [x1 x2 x3];
end

[~,PredictedGroup] = max([X1;X2;X3],[],2);
TrueGroup = [X1t; X2t; X3t];
C = confusionmat(TrueGroup,PredictedGroup);
heatmap(C, ...
    'YLabel', 'Actual condition', ...
    'YDisplayLabels', {'Healthy','Large Gap','Small Gap'}, ...
    'XLabel', 'Predicted condition', ...
    'XDisplayLabels', {'Healthy','Large Gap','Small Gap'}, ...
    'ColorbarVisible','off');
end