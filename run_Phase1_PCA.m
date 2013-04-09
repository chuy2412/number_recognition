%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Script to run PCA for number recognition
%Author: Jesus Rivera
%Inputs: 
%       set - number of sets to train
%Outputs:
%       Weight - the weight after training
%       Display the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set = 5000;
train = 3;
pc = 700;
%Train the network
trainNetwork_Phase1_PCA(set,train,pc)

 filename = ['db_set' int2str(set) '_train' int2str(train) '_PC_' ...
              int2str(pc) '.mat'];
filename

%Load the weight generated from the training phase
db = load(filename);

%Test results
testNetwork_Phase1_PCA(db.W_PCA,set)
