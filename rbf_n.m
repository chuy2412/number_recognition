%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train the network using RBF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
pixels = 28*28;
%Load data
I = load('mnist_all.mat');
X1  = reshape(I.train1(1,:),[],1);
X2  = reshape(I.train2(1,:),[],1);
X3  = reshape(I.train3(1,:),[],1);
X4  = reshape(I.train4(1,:),[],1);
X5  = reshape(I.train5(1,:),[],1);
X6  = reshape(I.train6(1,:),[],1);
X7  = reshape(I.train7(1,:),[],1);
X8  = reshape(I.train8(1,:),[],1);
X9  = reshape(I.train9(1,:),[],1);
X10 = reshape(I.train0(1,:),[],1);

PI = [X1 X2 X3 X4 X5 X6 X7 X8 X9 X10];

% fdim = [28 28];
% img = reshape(PI(:,1), fdim);
% img=rot90(img);
% img=rot90(img);
% img=rot90(img);
% img=fliplr(img);
% imshow(img)

%Inputs
P = [mean(X1) mean(X2) mean(X3) mean(X4)  mean(X5)...
     mean(X6) mean(X7) mean(X8) mean(X9) mean(X10)];
  
% % %Target 
T = [1 2 3 4 5 6 7 8 9 10];
 
%Train the network
net = newrb(P,T);
  
%Test the network
fprintf('Training the network using RBF\n')
for i=1:10
       p = mean(PI(:,i));
       Y = sim(net,p);
       fprintf('Test for %d is: %f (mean %f)\n',i,Y,p);
end
filename = 'rbf_net';
save(filename,'net');

