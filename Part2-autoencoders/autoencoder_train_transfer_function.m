load 'dataTest.mat';
load 'dataTrain.mat';

file_path = './Results/part1/';

encoder_functions = {'logsig', 'satlin'};
decoder_functions = {'logsig', 'satlin', 'purelin'};
    
hiddenSize1 = 100;
best_sreg = 10;
best_sprop = 0.1;

[~, e] = size(encoder_functions);
[~, d] = size(decoder_functions);
mse_errors = zeros(e, d);

epoch = 1000;
datatest_save = true;
out = 1;

for enc = encoder_functions
    in = 1;
    for dec = decoder_functions
        autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
            'EncoderTransferFunction', enc{1}, ...
            'DecoderTransferFunction', dec{1}, ...
            'MaxEpochs', epoch, ...
            'SparsityRegularization', best_sreg, ...
            'SparsityProportion', best_sprop);
        
        epoch_str = ['ep',num2str(epoch)];
        params = ['_', epoch_str, '_', ...
            'sr', num2str(best_sreg), '_', ...
            'sp', num2str(best_sprop), '_', ...
            'enc', enc{1}, '_', ...
            'dec', dec{1}];
        
        net_file_name = [file_path, 'network/', 'net', ...
            params ];
        save(net_file_name, 'autoenc1');
        
        fig1 = figure;
        plotWeights(autoenc1);
        weights_file_name = [file_path, 'weights/', 'weights', ...
             '_', params, '.jpg'];
        saveas(fig1, weights_file_name);
        
        reconstructed_fn = [file_path, 'reconstructed/', ...
                'reconstructed', params];
        reconstructed = predict(autoenc1, dataTest);
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
        
        in = in + 1;
    end
    out = out + 1;
end

disp(mse_errors);

error_txt = [file_path, 'errors/', 'mseerrors_transfer_functions', '.txt'];
fid = fopen(error_txt, 'wt');
for ii = 1:size(mse_errors,1)
    fprintf(fid,'%g\t',mse_errors(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);