function vary_lr(trainStore, testStore, layers, options)
    lr1 = 0.001;
    lr2 = 0.003;
    lr = zeros(8,1);
    accuracy = zeros(8,1);
    count = 1;
    while lr1 <= 1
        lr(count,1) = lr1;
        lr(count + 1,1) = lr2;
        accuracy(count,1) = train_with_varied_lr(trainStore, testStore, layers, options, lr1);
        accuracy(count + 1,1) = train_with_varied_lr(trainStore, testStore, layers, options, lr2);
        lr1 = 10*lr1;
        lr2 = 10*lr2;
        count = count + 2;
    end
    h = figure;
    plot(lr,accuracy);
    saveas(h,'.\plots\learning_rate\accuracy.jpg');
end

function accuracy = train_with_varied_lr(trainStore, testStore, layers, options, lr)
    options = trainingOptions('sgdm', ...
    'MaxEpochs', 25,...
    'InitialLearnRate',lr, ...
    'MiniBatchSize', 128, ...
    'Momentum', 0.1, ...
    'L2Regularization', 0.0001 ...
    );
    
    [convnet, tr] = trainNetwork(trainStore,layers,options);
    
    tempArr = size(trainStore.Files)/128;
    iterPerEpoch = floor(tempArr(1));
    iterArr = size(tr.TrainingAccuracy);
    iter = iterArr(2);
    epochs = floor(iter/iterPerEpoch);
    accuracyByEpoch = zeros(epochs);
    for i = 1:epochs
        accuracyByEpoch(i) = tr.TrainingAccuracy(i*iterPerEpoch);
    end
    
    lr_str = num2str(lr);
    file_name = strrep(lr_str,'.','_');
    h = figure;
    plot(accuracyByEpoch);
    saveas(h,['.\plots\learning_rate\accuracy\',file_name]);
        
    accuracy = find_accuracy(convnet, testStore);
    
end