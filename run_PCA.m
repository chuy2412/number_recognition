%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Script to run PCA for number recognition
%Author: Jesus Rivera
%Inputs: 
%       set - number of sets to train
%Outputs:
%       Weight - the weight after training
%       Display the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set = 1;

%Train the network
trainNetwork_PCA(set)

%Load the weight generated from the training phase
W = load('W_PCA.mat');

%Test results
testNetwork_PCA(W.W_PCA,set)
