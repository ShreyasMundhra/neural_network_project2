function save_all_plots(tr,net,P_train,T_train,file_dir,file_name)
    save_performance_plot(tr,file_dir,file_name);
    save_train_state_plot(tr,file_dir,file_name);
    save_err_histogram_plot(net,P_train,T_train,file_dir,file_name);
    save_regression_plot(net,P_train,T_train,file_dir,file_name);
end

function save_performance_plot(tr,file_dir,file_name)
    filename = [file_dir,'performance\',file_name];
    h = figure;
    plotperform(tr);
    saveas(h,filename,'jpg');
end

function save_train_state_plot(tr,file_dir,file_name)
    filename = [file_dir,'train_state\',file_name];
    h = figure;
    plottrainstate(tr);
    saveas(h,filename,'jpg');
end

function save_err_histogram_plot(net,P_train,T_train,file_dir,file_name)
    filename = [file_dir,'error_histogram\',file_name];
    h = figure;
    ploterrhist(T_train - net(P_train));
    saveas(h,filename,'jpg');
end

function save_regression_plot(net,P_train,T_train,file_dir,file_name)
    filename = [file_dir,'regression\',file_name];
    h = figure;
    plotregression(T_train,net(P_train));
    saveas(h,filename,'jpg');
end