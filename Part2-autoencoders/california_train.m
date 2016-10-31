load('cal_trained_net.mat');
P_train = P_train_std; T_train = T_train_std; Val = Val_std; %%% Use this line to use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
% hiddenLayerSize = [10];
% net = fitnet(hiddenLayerSize);
% net.trainFcn = 'trainlm';
% net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
% net.trainParam.epochs =200;
% net.trainParam.max_fail = 50;
vary_hidden_layers(net,P_train,T_train,Val.P,Val.T,TS_train_std);
% [net tr] = train(net,P_train,T_train);
% [fields N] = size(T_test);
% 
% est = net(Val.P);
% est = mapstd('apply', est, TS_train_std); %%% Use this line if you use STD or PCA preprocessing on the data. IMPORTANT: Uncomment the corresponding line above 
% 
% RMS_Error = perform(net, T_test, est);
% disp(RMS_Error);