P_train = P_train_std; 
T_train = T_train_std; Val = Val_std;

hiddenLayerSize = transpose([10,20,30]);
MS_Error = zeros(3,1);
for i = 1:size(hiddenLayerSize)
    net = fitnet(hiddenLayerSize(i));
    net.trainFcn = 'trainlm'; 
    net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0/100;
    net.trainParam.epochs = 200;
    net.trainParam.max_Fail = 25;
    [net tr] = train(net,P_train,T_train);
    [fields N] = size(T_test);
    
    est = net(Val.P);
    est = mapstd('apply', est, TS_train_std); %%% Use this line if you use STD or PCA preprocessing on the data. IMPORTANT: Uncomment the corresponding line above 
    MS_Error(i) = perform(net, T_test, est); % equivalent to sqrt(mean((T_test - est).^2));
    disp(num2str((MS_Error(i)),'%.10f'));
    savePerformancePlot(['Performance_hidlayer_size_',strrep(num2str(hiddenLayerSize(i)),'.','_')],tr);
    saveErrorHistogram(['ErrorHist_hidlayer_size_',strrep(num2str(hiddenLayerSize(i)),'.','_')],gsubtract(T_train,net(P_train)));
    saveTrainStatePlot(['TrainState_hidlayer_size_',strrep(num2str(hiddenLayerSize(i)),'.','_')],tr);
    saveRegressionPlot(['Reg_hidlayer_size_',strrep(num2str(hiddenLayerSize(i)),'.','_')],T_train,net(P_train));
end

% saveMissclassificationPlot('hiddenlayersize_MSError',hiddenLayerSize,MS_Error);

function savePerformancePlot(figureName,tr)
    fileName = ['plots\hidden_layers\performance\',figureName];
    h = figure;
    plotperform(tr);
    saveas(h,[fileName,'.jpg']);
end

function saveErrorHistogram(figureName,graphInput)
    fileName = ['plots\hidden_layers\error_histogram\',figureName];
    h = figure;
    ploterrhist(graphInput);
    saveas(h,[fileName,'.jpg']);
end

function saveRegressionPlot(figureName,T_train, output)
    fileName = ['plots\hidden_layers\regression\',figureName];
    h = figure;
    plotregression(T_train,output);
    saveas(h,[fileName,'.jpg']);
end

function saveTrainStatePlot(figureName,tr)
    fileName = ['plots\hidden_layers\train_state\',figureName];
    h = figure;
    plottrainstate(tr);
    saveas(h,[fileName,'.jpg']);
end

function saveMissclassificationPlot(figureName,epochs,misclassificationRate)
    fileName = ['plots\hidden_layers',figureName];
    h = figure;
    plot(epochs,misclassificationRate);
    saveas(h,[fileName,'.jpg']);
end