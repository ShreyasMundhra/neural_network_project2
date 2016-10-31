function create_net(hiddenLayerSize,P_train_std,T_train_std)
    net = fitnet(hiddenLayerSize);
    
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0;
    
    net.trainParam.epochs = 200;
    net.trainParam.max_fail = 50;

    net.trainFcn = 'traingd'; 
    
    numhidden = size(hiddenLayerSize);
    for i = 1:numhidden
        net.layers{i + 1}.transferFcn = 'tansig'; %Hidden layer function
    end
    
    net = configure(net,P_train_std,T_train_std);
    save('.\cal_trained_net.mat','net');
end