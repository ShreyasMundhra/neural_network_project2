function vary_batchsize(trainStore, testStore, layers, options)
    b = 16;
    batchSize = zeros(4,1);
    accuracy = zeros(4,1);
    count = 1;
    while b <= 128
        batchSize(count,1) = b;
        accuracy(count,1) = train_with_varied_batchsize(trainStore, testStore, layers, options, b);
        b = 2*b;
        count = count + 1;
    end
    h = figure;
    plot(batchSize,accuracy);
    saveas(h,'.\plots\batchsize\accuracy.jpg');
end

function accuracy = train_with_varied_batchsize(trainStore, testStore, layers, options, b)
    options = trainingOptions('sgdm', ...
        'MaxEpochs', 25,...
        'InitialLearnRate',0.001, ...
        'MiniBatchSize', b, ...
        'Momentum', 0.1, ...
        'L2Regularization', 0.0001 ...
        );
    [convnet, tr] = trainNetwork(trainStore,layers,options);
    
    tempArr = size(trainStore.Files)/b;
    iterPerEpoch = floor(tempArr(1));
    iterArr = size(tr.TrainingAccuracy);
    iter = iterArr(2);
    epochs = floor(iter/iterPerEpoch);
    accuracyByEpoch = zeros(epochs);
    for i = 1:epochs
        accuracyByEpoch(i) = tr.TrainingAccuracy(i*iterPerEpoch);
    end
    file_name = num2str(b);
    h = figure;
    plot(accuracyByEpoch);
    saveas(h,['.\plots\batchsize\accuracy\',file_name]);
    
    accuracy = find_accuracy(convnet, testStore);
end