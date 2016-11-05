% autoencoderPreprocess();

load 'dataTest.mat';
load 'dataTrain.mat';

%Take 100 samples for training and 20 for testing
% dataTestSubset = dataTest(1, 1:20);
% dataTestSubset = dataTest(1, :);
% dataTrainSubset = dataTrain(1, 1:100);
% dataTrainSubset = dataTrain(1, :);

hiddenSize1 = 100;

autoenc1 = trainAutoencoder(dataTrain,hiddenSize1);

figure(), plotWeights(autoenc1);

reconstructed = predict(autoenc1, dataTest);

mseError = 0;
for i = 1:numel(dataTest)
    mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
end

mseError = mseError/i;

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(dataTest{i});
end

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(reconstructed{i});
end

        % save('autoenc1_arch.png', autoenc1);
        % fig = figure();
        % savefig('autoenc1_fig.fig');
        % 
        % view(autoenc1)
        % savefig(fig, 'autoenc1_arch.fig');
        % 
        % clf(fig), plotWeights(autoenc1);
        % savefig(fig, 'autoenc1_weights.fig');

        % net = network(autoenc1);
        % [net, tr] = train(net);
        % clf(fig), plotperform(tr);
        % savefig(fig, 'autoenc_pp.fig');

        % figure(), plotWeights(autoenc1);
