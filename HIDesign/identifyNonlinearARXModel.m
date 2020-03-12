function syse = identifyNonlinearARXModel(Mmot,w,Q,Ts,N)
%identifyNonlinearARXModel 识别非线性的自回归各态历经模型对于双输入(w, Q), 单输出(Mmot)数据
% 输入：
%  w: 转速
%  Q: 液体流出率
%  Mmot: 电机转矩
%  N: 使用样本的个数
% 输出：
%  syse: 识别到的模型
%
% 本函数使用来自系统识别工具箱中的NLARX估计器

sys = idnlarx([2 2 1 0 1],'','CustomRegressors',{'u1(t-2)^2','u1(t)*u2(t-2)','u2(t)^2'});
data = iddata(Mmot,[w Q],Ts);
opt = nlarxOptions;
opt.Focus = 'simulation';
opt.SearchOptions.MaxIterations = 500;
syse = nlarx(data(1:N),sys,opt);
end