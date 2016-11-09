% Uncomment the code below if dataTrain and dataTest does not exist
% autoencoderPreprocess(); 
load 'dataTest.mat';
load 'dataTrain.mat';

file_path = './Results/part1/';

hiddenSize1 = 100;

sparsity_regs = [1, 4, 7, 10];
sparsity_props = [0.01, 0.05, 0.1, 0.2, 0.3];

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
        
        epoch_str = ['_', 'ep',num2str(epoch)];
        params = [epoch_str, '_', ...
            'sr', num2str(sparsity_regs(reg)), '_', ...
            'sp', num2str(sparsity_props(prop))];
        
        
        net_file_name = [file_path, 'network/', 'net', params, '.mat'];
        save(net_file_name, 'autoenc1');
        
        fig1 = figure;
        plotWeights(autoenc1);
        weights_file_name = [file_path, 'weights/', 'weights', ...
            params, '.jpg'];
        saveas(fig1, weights_file_name);
        
        
        reconstructed_file_name = [file_path, 'reconstructed/', ...
                'reconstructed', params, '.mat'];
        reconstructed = predict(autoenc1, dataTest);
        save(reconstructed_file_name, 'reconstructed');

        mseError = 0;
        for i = 1:numel(dataTest)
            mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
        end

        mseError = mseError/i;
        mse_errors(prop, reg) = mseError;
        disp(['mseError', params, ': ', num2str(mseError)]);
     
        fig3 = figure;
        for i = 1:20
            subplot(4,5,i);
            imshow(reconstructed{i}, [0 255]);
        end
        saveas(fig3, [reconstructed_file_name, '.jpg']);
        
    end
end

disp(mse_errors);

error_txt = [file_path, 'errors/', 'mseerrors_sparsity', '.txt'];
fid = fopen(error_txt, 'wt');
for ii = 1:size(mse_errors,1)
    fprintf(fid,'%g\t',mse_errors(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);