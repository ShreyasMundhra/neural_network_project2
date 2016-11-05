load 'dataTest.mat';
load 'dataTrain.mat';

file_path = './Results/part2/';
    
hiddenSize1 = 100;
hiddenSize2 = 50;
encoder_function = 'logsig';
decoder_function = 'logsig';
sreg = 10;
sprop = 0.1;
epoch = 1000;

epoch_str = ['ep',num2str(epoch)];
params = ['_second_', epoch_str, '_', ...
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
view(autoenc1);

net_file_name = [file_path, 'network/', 'net1', ...
    params1 ];
save(net_file_name, 'autoenc1');

fig1 = figure;
plotWeights(autoenc1);
weights_file_name = [file_path, 'weights/', 'weights1', ...
     '_', params1, '.jpg'];
saveas(fig1, weights_file_name);

features1 = encode(autoenc1, dataTrain);
features1_fn = [file_path, 'features/', 'encoded1', params1, '.mat'];
save(features1_fn, 'features1');

autoenc2 = trainAutoencoder(features1, hiddenSize2,...
    'EncoderTransferFunction', encoder_function, ...
    'DecoderTransferFunction', decoder_function, ...
    'MaxEpochs', epoch, ...
    'SparsityRegularization', sreg, ...
    'SparsityProportion', sprop);
view(autoenc2);

features2 = encode(autoenc2, features1);
features2_fn = [file_path, 'features/', 'encoded2', params2, '.mat'];
save(features2_fn, 'features2');

net_file_name = [file_path, 'network/', 'net2', ...
    params2, '.mat' ];
save(net_file_name, 'autoenc2');

fig2 = figure;
plotWeights(autoenc2);
weights_file_name = [file_path, 'weights/', 'weights2', ...
     '_', params2, '.jpg'];
saveas(fig2, weights_file_name);

feats1 = encode(autoenc1, dataTest);
feats1_fn = [file_path, 'features/', 'encodedTest', params1, '.mat'];
save(feats1_fn, 'feats1');

feats2 = encode(autoenc2, feats1);
feats2_fn = [file_path, 'features/', 'encodedTest', params2, '.mat'];
save(feats2_fn, 'feats2');

dec2 = decode(autoenc2, feats2);
dec2_fn = [file_path, 'decoded/', 'decodedTest', params2, '.mat'];
save(dec2_fn, 'dec2');

reconstructed = decode(autoenc1, dec2);
reconstructed_fn = [file_path,  'decoded/', 'reconstructed_second_', ...
    params, '.mat'];
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
