load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTrain.mat';
load 'labelsTest.mat';

file_path = './Results/part3/';
    
% hiddenSize1 = 100;
% hiddenSize2 = 50;
hiddenSize1 = 300;
hiddenSize2 = 100;

% hiddenSizes1 = [100, 300, 500, 700];
% hiddenSizes2 = [50, 100, 200];
encoder_function = 'logsig';
decoder_function = 'logsig';
% encoder_functions = {'logsig', 'satlin'};
% decoder_functions = {'logsig', 'satlin', 'purelin'};
sreg = 10;
sprop = 0.1;
% sparsity_regs = [1, 4, 7, 10];
% sparsity_props = [0.1, 0.2, 0.3];
epoch = 1;

% for r = 1:numel(sparsity_regs)
%     for p = 1:numel(sparsity_props)
        epoch_str = ['ep',num2str(epoch)];
        params = ['_test_', epoch_str, '_', ...
            'h1', num2str(hiddenSize1), ...
            'h2', num2str(hiddenSize2), ...
            'sr', num2str(sreg), '_', ...
            'sp', num2str(sprop), '_', ...
            'enc', encoder_function, '_', ...
            'dec', decoder_function];
        params1 = [params, '_features1'];
        params2 = [params, '_features2'];

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
        features1_fn = [file_path, 'features/', 'autoenc1', params1];
        save(features1_fn, 'features1');

        autoenc2 = trainAutoencoder(features1, hiddenSize2,...
            'EncoderTransferFunction', encoder_function, ...
            'DecoderTransferFunction', decoder_function, ...
            'MaxEpochs', epoch, ...
            'SparsityRegularization', sreg, ...
            'SparsityProportion', sprop);
        view(autoenc2);

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
        disp(['mseError', params, ': ', num2str(mseError)]);

        fig3 = figure;
        for i = 1:20
            subplot(4,5,i);
            imshow(reconstructed{i}, [0 255]);
        end
        saveas(fig3, [reconstructed_fn, '.jpg']);

        features2 = encode(autoenc2, features1);
        softnet = trainSoftmaxLayer(features2, labelsTrain, ...
            'LossFunction', 'crossentropy', ...
            'MaxEpochs', epoch); 
        view(softnet);

        y = softnet(feats2);

        fig4 = figure;
        plotconfusion(labelsTest, y);
        confusion_file_name = [file_path, 'confusion/', ...
                'autoenc2_confusion_', params, '.jpg'];
        saveas(fig4, confusion_file_name);
%     end
% end

% error_txt = [file_path, 'errors/', 'mseerrors_third_pure_hiddens', '.txt'];
% fid = fopen(error_txt, 'wt');
% for ii = 1:size(mse_errors,1)
%     fprintf(fid,'%g\t',mse_errors(ii,:));
%     fprintf(fid,'\n');
% end
% fclose(fid);