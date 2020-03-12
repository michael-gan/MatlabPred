function Qest = simulatePumpPipeModel(Ts,th3,th4,th5)
% 动态管道系统的分段线性模型
% Ts: 采样时间
% w: 水泵转速
% th1, th2, th3 对于三种转速区间估计得到的参数，每个参数形状为3*1.
% 本函数要求安装控制系统工具箱.

ss1 = ss(th5(1),th4(1),th5(1),th4(1),Ts);
ss2 = ss(th5(2),th4(2),th5(2),th4(2),Ts);
ss3 = ss(th5(3),th4(3),th5(3),th4(3),Ts);
offset = permute([th3(1),th3(2),th3(3)]',[3 2 1]);
OP = struct('Region',[1 2 3]');
sys = cat(3,ss1,ss2,ss3);
sys.SamplingGrid = OP;

assignin('base','sys',sys)
assignin('base','offset',offset)
mdl = 'LPV_pump_pipe';
sim(mdl);
Qest = logsout.get('Qest');
Qest = Qest.Values;
Qest = Qest.Data;
end