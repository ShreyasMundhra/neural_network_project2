% autoencoderPreprocess();

load 'dataTest.mat';
load 'dataTrain.mat';

file_path = './Results/';

hiddenSize1 = 100;

epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
[~, p] = size(epochs);
mse_errors = zeros(p);

for ep = 1:numel(epochs)
    params = ['_', 'epoch', num2str(epochs(ep))];
    
    autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
        'MaxEpochs', epochs(ep));
    
    f1 = figure();
    weights_file_name = [file_path, 'weights/', 'autoenc1_weights', ...
        params, '.jpg'];
    plotWeights(autoenc1);
    saveas(f1, weights_file_name);

    reconstructed = predict(autoenc1, dataTest);

    mseError = 0;
    for i = 1:numel(dataTest)
        mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
    end

    mseError = mseError/i;
    mse_errors(ep) = mseError;
    
    disp(['mseError', params, ': ', num2str(mseError)]);

    f2 = figure();
    for i = 1:20
        subplot(4,5,i);
        imshow(dataTest{i});
    end
    dataset_file_name = [file_path, 'datatest/', ...
            'autoenc1_datatest', params, '.jpg'];
    saveas(f2, dataset_file_name);

    f3 = figure();
    for i = 1:20
        subplot(4,5,i);
        imshow(reconstructed{i});
    end
    reconstructed_file_name = [file_path, 'reconstructed/', ...
            'autoenc1_reconstructed', params, '.jpg'];
    saveas(f3, reconstructed_file_name);
end

error_txt = [file_path, 'errors/', 'autoenc1_epoch', '.txt'];
fid = fopen(error_txt, 'wt');
for ii = 1:size(mse_errors,1)
    fprintf(fid,'%g\t',mse_errors(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);