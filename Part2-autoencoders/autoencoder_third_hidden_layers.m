load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTrain.mat';
load 'labelsTest.mat';

file_path = './Results/part3/';

hiddenSizes1 = [400, 300, 200, 100];
hiddenSizes2 = [200, 100, 50];

[~, e] = size(hiddenSizes1);
[~, d] = size(hiddenSizes2);
mse_errors = zeros(e, d);

encoder_function = 'logsig';
decoder_function = 'logsig';

sreg = 10;
sprop = 0.1;

epoch = 200;

for h1 = 1:numel(hiddenSizes1)
    epoch_str = ['ep',num2str(epoch)];
    
    params = ['_hidden_', epoch_str, '_', ...
            'sr', num2str(sreg), '_', ...
            'sp', num2str(sprop), '_', ...
            'enc', encoder_function, '_', ...
            'dec', decoder_function, '_', ...
            'h1', num2str(hiddenSizes1(h1))];
        
    params1 = [params, '_autoenc1'];
        
    autoenc1 = trainAutoencoder(dataTrain, hiddenSizes1(h1), ...
            'EncoderTransferFunction', encoder_function, ...
            'DecoderTransferFunction', decoder_function, ...
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
    features1_fn = [file_path, 'features/', 'autoenc1', params1];
    save(features1_fn, 'features1');
    
    feats1 = encode(autoenc1, dataTest);

    for h2 = 1:numel(hiddenSizes2)
        params2 = [params, 'h2', num2str(hiddenSizes2(h2)), '_autoenc2'];

        autoenc2 = trainAutoencoder(features1, hiddenSizes2(h2),...
            'EncoderTransferFunction', encoder_function, ...
            'DecoderTransferFunction', decoder_function, ...
            'MaxEpochs', epoch, ...
            'SparsityRegularization', sreg, ...
            'SparsityProportion', sprop);
        
        features2 = encode(autoenc2, features1);
        features2_fn = [file_path, 'features/', params2];
        save(features2_fn, 'features2');
        
        net_file_name = [file_path, 'network/', 'net2', ...
            params2 ];
        save(net_file_name, 'autoenc2');

        fig2 = figure;
        plotWeights(autoenc2);
        weights_file_name = [file_path, 'weights/', 'weights2', ...
             '_', params2, '.jpg'];
        saveas(fig2, weights_file_name);

        feats2 = encode(autoenc2, feats1);
        dec2 = decode(autoenc2, feats2);
        reconstructed = decode(autoenc1, dec2);
        reconstructed_fn = [file_path, 'decoded/', ...
            'reconstructed',  params2, '.mat'];
        save(reconstructed_fn, 'reconstructed');

        mseError = 0;
        for i = 1:numel(dataTest)
            mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
        end

        mseError = mseError/i;
        disp(['mseError', params2, ': ', num2str(mseError)]);
        mse_errors(h1, h2) = mseError;

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
            'softnet', params2, '.mat'];
        save(predicted_file_name, 'y');

        fig4 = figure;
        plotconfusion(labelsTest, y);
        confusion_file_name = [file_path, 'confusion/', ...
                'confusion_', params2, '.jpg'];
        saveas(fig4, confusion_file_name);
    end
end

error_txt = [file_path, 'errors/', 'mseerrors_third_hidden', '.txt'];
fid = fopen(error_txt, 'wt');
for ii = 1:size(mse_errors,1)
    fprintf(fid,'%g\t',mse_errors(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);