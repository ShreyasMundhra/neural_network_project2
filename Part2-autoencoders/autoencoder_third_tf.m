load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTrain.mat';
load 'labelsTest.mat';

file_path = './Results/part3/';
    
hiddenSize1 = 300;
hiddenSize2 = 100;

hiddenSizes1 = [100, 300, 500, 700];
hiddenSizes2 = [50, 100, 200];
encoder_functions = {'logsig', 'satlin'};
decoder_functions = {'logsig', 'satlin', 'purelin'};

[~, e] = size(encoder_functions);
[~, d] = size(decoder_functions);
mse_errors = zeros(e, d);

sreg = 10;
sprop = 0.1;
sparsity_regs = [1, 5, 10];
sparsity_props = [0.1, 0.2, 0.3];
epoch = 200;

out = 1;
for ef = 1:numel(encoder_functions)
    in = 1;
    for df = 1:numel(decoder_functions)
        epoch_str = ['ep',num2str(epoch)];
        params = ['_tf_', epoch_str, '_', ...
            'h1', num2str(hiddenSize1), ...
            'h2', num2str(hiddenSize2), ...
            'sr', num2str(sreg), '_', ...
            'sp', num2str(sprop), '_', ...
            'enc', encoder_functions(ef), '_', ...
            'dec', decoder_functions(df)];
        params1 = [params, '_features1'];
        params2 = [params, '_features2'];

        autoenc1 = trainAutoencoder(dataTrain, hiddenSizes1(h1), ...
            'EncoderTransferFunction', encoder_functions(ef), ...
            'DecoderTransferFunction', decoder_functions(df), ...
            'MaxEpochs', epoch, ...
            'SparsityRegularization', sreg, ...
            'SparsityProportion', sprop);

        net_file_name = [file_path, 'network/', 'net1', ...
            params1 ];
        save(net_file_name, 'autoenc1');

        fig1 = figure;
        plotWeights(autoenc1);
        weights_file_name = [file_path, 'weights/', 'weights1', ...
             '_', params1, '.jpg'];
        saveas(fig1, weights_file_name);

        features1 = encode(autoenc1, dataTrain);
        eatures1_fn = [file_path, 'features/', 'autoenc1', params1];
        save(features1_fn, 'features1');

        autoenc2 = trainAutoencoder(features1, hiddenSizes2(h2),...
            'EncoderTransferFunction', encoder_functions(ef), ...
            'DecoderTransferFunction', decoder_functions(df), ...
            'MaxEpochs', epoch, ...
            'SparsityRegularization', sreg, ...
            'SparsityProportion', sprop);

        net_file_name = [file_path, 'network/', 'net2', ...
            params2 ];
        save(net_file_name, 'autoenc2');

        fig2 = figure;
        plotWeights(autoenc2);
        weights_file_name = [file_path, 'weights/', 'weights2', ...
             '_', params2, '.jpg'];
        saveas(fig2, weights_file_name);

        feats1 = encode(autoenc1, dataTest);
        feats2 = encode(autoenc2, feats1);
        dec2 = decode(autoenc2, feats2);
        reconstructed = decode(autoenc1, dec2);
        reconstructed_fn = [file_path, 'decoded/', 'reconstructed_third_', params];
        save(reconstructed_fn, 'reconstructed');

        mseError = 0;
        for i = 1:numel(dataTest)
            mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
        end

        mseError = mseError/i;
        mse_errors(out, in) = mseError;
        disp(['mseError', params, ': ', num2str(mseError)]);

        fig3 = figure;
        for i = 1:20
            subplot(4,5,i);
            imshow(reconstructed{i}, [0 255]);
        end
        saveas(fig3, [reconstructed_fn, '.jpg']);


        softnet = trainSoftmaxLayer(feats2, labelsTest, ...
            'LossFunction', 'crossentropy', ...
            'MaxEpochs', epoch); 

        y = softnet(feats2);
        predicted_file_name = [file_path, 'output', ...
            'softnet', params, '.jpg'];
        save(predicted_file_name, 'y');

        fig4 = figure;
        plotconfusion(labelsTest, y);
        confusion_file_name = [file_path, 'confusion/', ...
                'autoenc2_confusion_', params, '.jpg'];
        saveas(fig4, confusion_file_name);
        in = in + 1;
    end
    out = out + 1;
end

error_txt = [file_path, 'errors/', 'mseerrors_third_pure_hiddens', '.txt'];
fid = fopen(error_txt, 'wt');
for ii = 1:size(mse_errors,1)
    fprintf(fid,'%g\t',mse_errors(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);