load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTrain.mat';
load 'labelsTest.mat';

file_path = './Results/part3/';
    
hiddenSize1 = 300;
hiddenSize2 = 100;
encoder_function = 'logsig';
decoder_function = 'logsig';
sreg = 10;
sprop = 0.3;
epoch = 200;

epoch_str = ['ep',num2str(epoch)];
params = ['_stacked_', epoch_str, '_', ...
    'h1', num2str(hiddenSize1), '_', ...
    'h2', num2str(hiddenSize2), '_', ...
    'sr', num2str(sreg), '_', ...
    'sp', num2str(sprop), '_', ...
    'enc', encoder_function, '_', ...
    'dec', decoder_function];
params1 = [params, '_autoenc1'];
params2 = [params, '_autoenc2'];

autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
    'EncoderTransferFunction', encoder_function, ...
    'DecoderTransferFunction', decoder_function, ...
    'MaxEpochs', epoch, ...
    'SparsityRegularization', sreg, ...
    'SparsityProportion', sprop);

% view(autoenc1);
net_file_name = [file_path, 'network/', 'net1', ...
    params1 ];
save(net_file_name, 'autoenc1');

fig1 = figure;
plotWeights(autoenc1);
weights_file_name = [file_path, 'weights/', 'weights1', ...
     '_', params1, '.jpg'];
saveas(fig1, weights_file_name);

features1 = encode(autoenc1, dataTrain);
features1_fn = [file_path, 'features/', params1];
save(features1_fn, 'features1');

autoenc2 = trainAutoencoder(features1, hiddenSize2,...
    'EncoderTransferFunction', encoder_function, ...
    'DecoderTransferFunction', decoder_function, ...
    'MaxEpochs', epoch, ...
    'SparsityRegularization', sreg, ...
    'SparsityProportion', sprop);
% view(autoenc2);

net_file_name = [file_path, 'network/', 'autoenc1_net', ...
    params2 ];
save(net_file_name, 'autoenc2');

fig2 = figure;
plotWeights(autoenc2);
weights_file_name = [file_path, 'weights/', 'autoenc2_weights', ...
     '_', params2, '.jpg'];
saveas(fig2, weights_file_name);

features2 = encode(autoenc2, features1);
features2_fn = [file_path, 'features2', params2];
save(features2_fn, 'features2');

feats1 = encode(autoenc1, dataTest);
feats2 = encode(autoenc2, feats1);
dec2 = decode(autoenc2, feats2);
reconstructed = decode(autoenc1, dec2);
reconstructed_fn = [file_path, 'decoded/', ...
    'reconstructed_', params, '.mat'];
save(reconstructed_fn, 'reconstructed');

mseError = 0;
for i = 1:numel(dataTest)
    mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
end

mseError = mseError/i;
disp(['mseError', params, ': ', num2str(mseError)]);

fig3 = figure;
for i = 1:20
    subplot(4,5,i);
    imshow(reconstructed{i}, [0 255]);
end
saveas(fig3, [reconstructed_fn, '.jpg']);

softnet = trainSoftmaxLayer(features2, labelsTrain, ...
    'LossFunction', 'crossentropy', ...
    'MaxEpochs', epoch); 
deepnet = stack(autoenc1, autoenc2, softnet);
% view(softnet);
% view(deepnet);

%%%%%%%%%%%%%%%%%%%%Test%%%%%%%%%%%%%%%%%%%%%%%%
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Turn the test images into vectors and put them in a matrix
xTrain = zeros(inputSize, numel(dataTrain));
for i = 1:numel(dataTrain)
    xTrain(:,i) = dataTrain{i}(:);
end

deepnet = train(deepnet, xTrain, labelsTrain);
% view(deepnet);

xTest = zeros(inputSize, numel(dataTest));
for i = 1:numel(dataTest)
    xTest(:,i) = dataTest{i}(:);
end

y = deepnet(xTest);
predicted_file_name = [file_path, 'output/', ...
            'deepnet', params, '.mat'];
save(predicted_file_name, 'y');

fig4 = figure;
plotconfusion(labelsTest, y);
confusion_file_name = [file_path, 'confusion/', ...
                'confusion_', params, '.jpg'];
saveas(fig4, confusion_file_name);
