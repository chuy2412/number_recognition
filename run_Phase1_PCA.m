%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Script to run PCA for number recognition
%Author: Jesus Rivera
%Inputs: 
%       set - number of sets to train
%Outputs:
%       Weight - the weight after training
%       Display the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %For training and testing
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
set = 1;
train = 3;
pc = 700;
repeat = 10;
cycle = 100;

filename = ['db_spiral_cycle_' int2str(cycle) '_set_' int2str(set)... 
            '_repeat_' int2str(repeat) '_train' int2str(train)... 
            '_PC_' int2str(pc)];
        
%Train network
trainNetwork_Phase1_PCA(cycle,set,repeat,train,pc,filename)

% %Load the weight generated from the training phase
file_name = [filename '.mat']
db = load(file_name);

%Test results
testNetwork_Phase1_PCA(db.W_PCA,set)


% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %For just testing
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Provide filename
% filename = 'db_spiral_cycle_20_set_1_repeat_15_train3_PC_700.mat';
% db = load(filename);
% 
% set = 5000;
% %Test results
% testNetwork_Phase1_PCA(db.W_PCA,set);
