% Uncomment the code below if dataTrain and dataTest does not exist
% autoencoderPreprocess();load 'dataTest.mat';

load 'dataTrain.mat';
load 'labelsTrain.mat';
load 'labelsTest.mat';

file_path = './Results/part3/';

hiddenSize1 = 300;
hiddenSize2 = 100;

sparsity_regs = [1, 4, 7, 10];
sparsity_props = [0.1, 0.2, 0.3];

[~, e] = size(sparsity_regs);
[~, d] = size(sparsity_props);
mse_errors = zeros(e, d);

encoder_function = 'logsig';
decoder_function = 'logsig';

epoch = 200;

for r = 1:numel(sparsity_regs)
    for p = 1:numel(sparsity_props)
        epoch_str = ['ep',num2str(epoch)];
        params = ['_sparsity_', epoch_str, '_', ...
            'h1', num2str(hiddenSize1), ...
            'h2', num2str(hiddenSize2), ...
            'sr', num2str(sparsity_regs(r)), '_', ...
            'sp', num2str(sparsity_props(p)), '_', ...
            'enc', encoder_function, '_', ...
            'dec', decoder_function];
        params1 = [params, '_autoenc1'];
        params2 = [params, '_autoenc2'];

        autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
            'EncoderTransferFunction', encoder_function, ...
            'DecoderTransferFunction', decoder_function, ...
            'MaxEpochs', epoch, ...
            'SparsityRegularization', sparsity_regs(r), ...
            'SparsityProportion', sparsity_props(p));

        net_file_name = [file_path, 'network/', 'net1', ...
            params1, '.mat'];
        save(net_file_name, 'autoenc1');

        fig1 = figure;
        plotWeights(autoenc1);
        weights_file_name = [file_path, 'weights/', 'weights1', ...
             params1, '.jpg'];
        saveas(fig1, weights_file_name);

        features1 = encode(autoenc1, dataTrain);
        features1_fn = [file_path, 'features/', 'features1', params1];
        save(features1_fn, 'features1');

        autoenc2 = trainAutoencoder(features1, hiddenSize2,...
            'EncoderTransferFunction', encoder_function, ...
            'DecoderTransferFunction', decoder_function, ...
            'MaxEpochs', epoch, ...
            'SparsityRegularization', sparsity_regs(r), ...
            'SparsityProportion', sparsity_props(p));

        net_file_name = [file_path, 'network/', 'net2', ...
            params2 ];
        save(net_file_name, 'autoenc2');

        features2 = encode(autoenc2, features1);
        features2_fn = [file_path, 'features/', 'features2', params2];
        save(features2_fn, 'features2');

        fig2 = figure;
        plotWeights(autoenc2);
        weights_file_name = [file_path, 'weights/', 'weights2', ...
            params2, '.jpg'];
        saveas(fig2, weights_file_name);

        feats1 = encode(autoenc1, dataTest);
        feats2 = encode(autoenc2, feats1);
        dec2 = decode(autoenc2, feats2);
        reconstructed = decode(autoenc1, dec2);
        reconstructed_fn = [file_path, 'reconstructed/', ...
            'reconstructed', params, '.mat'];
        save(reconstructed_fn, 'reconstructed');

        mseError = 0;
        for i = 1:numel(dataTest)
            mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
        end

        mseError = mseError/i;
        disp(['mseError', params, ': ', num2str(mseError)]);
        mse_errors(r, p) = mseError;

        fig3 = figure;
        for i = 1:20
            subplot(4,5,i);
            imshow(reconstructed{i}, [0 255]);
        end
        saveas(fig3, [reconstructed_fn, '.jpg']);

        softnet = trainSoftmaxLayer(features2, labelsTrain, ...
            'LossFunction', 'crossentropy', ...
            'MaxEpochs', epoch); 

        y = softnet(feats2);
        predicted_file_name = [file_path, 'output/', ...
            'softnet', params, '.mat'];
        save(predicted_file_name, 'y');

        fig4 = figure;
        plotconfusion(labelsTest, y);
        confusion_file_name = [file_path, 'confusion/', ...
                'confusion', params, '.jpg'];
        saveas(fig4, confusion_file_name);
    end
end

error_txt = [file_path, 'errors/', 'mseerrors_third_pure_sparsity', '.txt'];
fid = fopen(error_txt, 'wt');
for ii = 1:size(mse_errors,1)
    fprintf(fid,'%g\t',mse_errors(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);