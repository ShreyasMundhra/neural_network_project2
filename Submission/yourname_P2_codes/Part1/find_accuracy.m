function accuracy = find_accuracy(convnet, testStore)
    YTest = classify(convnet, testStore);
    TTest = testStore.Labels;
    accuracy = sum(YTest == TTest)/numel(YTest);
end