%cnnPreprocess('..\Images_Data_Clipped');

load 'dataTeststore.mat';
load 'dataTrainstore.mat';

imageDim = 28;

layers = [imageInputLayer([imageDim imageDim]), ...
    convolution2dLayer([5, 5],30), ...
    averagePooling2dLayer(2), ...
    convolution2dLayer([5, 5],50), ...
    averagePooling2dLayer(2), ...
    fullyConnectedLayer(10), ...
    softmaxLayer(), ...
    classificationLayer()];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 20,...
    'InitialLearnRate',0.001, ...
    'MiniBatchSize', 128, ...
    'Momentum', 0.1, ...
    'L2Regularization', 0.001 ...
    );

vary_num_filters(dataTrainstore,dataTeststore,layers,options);

% Train with the optimum parameter values and find accuracy on test set
layers(2) = convolution2dLayer([5,5],50);
layers(4) = convolution2dLayer([5,5],60);
[convnet, tr] = trainNetwork(dataTrainstore,layers,options);
tempArr = size(dataTrainstore.Files)/128;
iterPerEpoch = floor(tempArr(1));
iterArr = size(tr.TrainingAccuracy);
iter = iterArr(2);
epochs = floor(iter/iterPerEpoch);
accuracyByEpoch = zeros(epochs);
for i = 1:epochs
    accuracyByEpoch(i) = tr.TrainingAccuracy(i*iterPerEpoch);
end
h = figure;
plot(accuracyByEpoch);
saveas(h,'.\plots\accuracyPart2');
accuracy = find_accuracy(convnet, dataTeststore);
disp('Accuracy');
disp(accuracy);