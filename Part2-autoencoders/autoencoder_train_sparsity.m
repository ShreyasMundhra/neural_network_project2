load 'dataTest.mat';
load 'dataTrain.mat';

file_path = './Results/';

hiddenSize1 = 100;

sparsity_regs = [1,  4,  7, 10];
sparsity_props = [0.1, 0.15, 0.2, 0.25, 0.3];

reg = 1;  
prop = 1;

[~, p] = size(sparsity_props);
[~, r] = size(sparsity_regs);
mse_errors = zeros(p, r);

epoch = 1000;

for prop = 1:numel(sparsity_props)
    for reg = 1:numel(sparsity_regs)
        autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
            'MaxEpochs', epoch, ...
            'SparsityRegularization', sparsity_regs(reg), ...
            'SparsityProportion', sparsity_props(prop));
        
        epoch_str = ['ep',num2str(epoch)];
        params = ['_', epoch_str ...
            'sr', num2str(sparsity_regs(reg)), '_', ...
            'sp', num2str(sparsity_props(prop))];
        
        
        net_file_name = [file_path, 'network/', 'autoenc1_net', ...
            params ];
        save(net_file_name, 'autoenc1');
        
        fig1 = figure;
        plotWeights(autoenc1);
        weights_file_name = [file_path, 'weights/', 'autoenc1_weights', ...
             '_', params, '.jpg'];
        saveas(fig1, weights_file_name);
        
        reconstructed = predict(autoenc1, dataTest);

        mseError = 0;
        for i = 1:numel(dataTest)
            mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
        end

        mseError = mseError/i;
        mse_errors(prop, reg) = mseError;
        disp(['mseError', params, ': ', num2str(mseError)]);

        fig2 = figure;
        for i = 1:20
            subplot(4,5,i);
            imshow(dataTest{i});
        end
        datatest_file_name = [file_path, 'datatest/', ...
                'autoenc1_datatest', params, num2str(i), '.jpg'];
        saveas(fig2, datatest_file_name);
     
        fig3 = figure;
        for i = 1:20
            subplot(4,5,i);
            imshow(reconstructed{i});
        end
        reconstructed_file_name = [file_path, 'reconstructed/', ...
                'autoenc1_reconstructed', params, '.jpg'];
        saveas(fig3, reconstructed_file_name);
        
    end
end

disp(mse_errors);

error_txt = ['./Results/errors/', 'autoenc1_sparsity', '.txt'];
fid = fopen(error_txt, 'wt');
for ii = 1:size(mse_errors,1)
    fprintf(fid,'%g\t',mse_errors(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);