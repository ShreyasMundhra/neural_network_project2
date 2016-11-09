function vary_num_filters(trainStore, testStore, layers, options)
    accuracy = zeros(6,6);
    for i = 20:10:70
        for j = 20:10:70
            layers(2) = convolution2dLayer([5,5],i);
            layers(4) = convolution2dLayer([5,5],j);
            accuracy((i/10)-1,(j/10)-1) = train_with_varied_filters(trainStore, testStore, layers, options,i,j);
        end
    end
    disp(accuracy);
end

function accuracy = train_with_varied_filters(trainStore, testStore, layers, options,layer1,layer2)
    [convnet, tr] = trainNetwork(trainStore,layers,options);
    
    tempArr = size(trainStore.Files)/64;
    iterPerEpoch = floor(tempArr(1));
    iterArr = size(tr.TrainingAccuracy);
    iter = iterArr(2);
    epochs = floor(iter/iterPerEpoch);
    accuracyByEpoch = zeros(epochs);
    for i = 1:epochs
        accuracyByEpoch(i) = tr.TrainingAccuracy(i*iterPerEpoch);
    end
    file_name = [num2str(layer1),'_',num2str(layer2)];
    h = figure;
    plot(accuracyByEpoch);
    saveas(h,['.\plots\num_filters\accuracy\',file_name]);
    
    accuracy = find_accuracy(convnet, testStore);
end