function vary_momentum_decay(trainStore, testStore, layers, options)
    m = 0.1;
    accuracy = zeros(3,8);

    for i = 1:3
        d1 = 0.0001;
        d2 = 0.0005;

        for j = 1:4
            accuracy(i,2*j-1) = train_with_varied_momentum_decay(trainStore, testStore, layers, options, m, d1);
            accuracy(i,2*j) = train_with_varied_momentum_decay(trainStore, testStore, layers, options, m, d2);
            d1 = 10*d1;
            d2 = 10*d2;
        end
        m = 3*m;
    end

    disp(accuracy);
end

function accuracy = train_with_varied_momentum_decay(trainStore, testStore, layers, options, m, d)
    options = trainingOptions('sgdm', ...
    'MaxEpochs', 25,...
    'InitialLearnRate',0.001, ...
    'MiniBatchSize', 128, ...
    'Momentum', m, ...
    'L2Regularization', d ...
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
    m_str = num2str(m);
    d_str = num2str(d);
    file_name = ['m',strrep(m_str,'.','_'),'d',strrep(d_str,'.','_')];
    h = figure;
    plot(accuracyByEpoch);
    saveas(h,['.\plots\momentum_decay\accuracy\',file_name]);
    
    accuracy = find_accuracy(convnet, testStore);
end