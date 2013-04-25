function trainNetwork_Phase1_PCA(cycle,set,repeat,train, pc,filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training Phase
%Author: Jesus Rivera
%This function trains the network for number recognition using PCA
%Then the Weight is store in a database so that it can be used for the
%Training phase
%Values that can be edited to improve performance of PCA
%Input:
%   set   - number of training sets
%   train - number of training steps per set
%   pc    - principal components
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fdim = [28,28];        %Dimentsion to show the image
m = 28*28;             %m dimension
l = pc;                  %l the number of principal components
%set=100 %4200;
W_PCA = 0.01*rand(l,m);
%W = 0.0001/m*rand(l,m);
Y = double(zeros(l));    
%Range of random values
rand_min=1;
rand_max=10; 
 % eta = 0.0065/(m); 
eta_PCA = 0.0065/m;  
 %   eta = (0.00001/(m))/((set/rand_max));
 
 %filename = ['db_set' int2str(set) '_train' int2str(train) ...
 %            '_PC_' int2str(pc)];

if set>5800
    set=5800;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  


dimension = [28,28];
pixels = m;
newdim = [1,pixels];

%Load data
I = load('mnist_all.mat');
XI = uint8(zeros(10,set,pixels));
XI(1,:,:) = I.train1(1:set,:);
XI(2,:,:) = I.train2(1:set,:);
XI(3,:,:) = I.train3(1:set,:);
XI(4,:,:) = I.train4(1:set,:);
XI(5,:,:) = I.train5(1:set,:);
XI(6,:,:) = I.train6(1:set,:);
XI(7,:,:) = I.train7(1:set,:);
XI(8,:,:) = I.train8(1:set,:);
XI(9,:,:) = I.train9(1:set,:);
XI(10,:,:) = I.train0(1:set,:);

X = double(zeros(m));

%Calculate the mean of each set
M0 = reshape(mean(XI(10,1:set,:),2),[],1);
M1 = reshape(mean(XI(1,1:set,:),2),[],1);
M2 = reshape(mean(XI(2,1:set,:),2),[],1);
M3 = reshape(mean(XI(3,1:set,:),2),[],1);
M4 = reshape(mean(XI(4,1:set,:),2),[],1);
M5 = reshape(mean(XI(5,1:set,:),2),[],1);
M6 = reshape(mean(XI(6,1:set,:),2),[],1);
M7 = reshape(mean(XI(7,1:set,:),2),[],1);
M8 = reshape(mean(XI(8,1:set,:),2),[],1);
M9 = reshape(mean(XI(9,1:set,:),2),[],1);


%Calculate the total mean
Xmean = mean((M0 + M1 + M2 + M3 + M4 + M5 + M6 +M7 +M8 + M9),2);
%newMean = mean((M1 +M2),2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Scale the mean so that the values are from 0-255
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MeanX = uint8(Xmean);
minimum = min(Xmean);  %get minimum
maximum = max(Xmean);  %get maximum
for i=1:m
    %Normalize pixel from 0-255
    MeanX(i,1) = 255*(Xmean(i) -minimum)/(maximum-minimum);
end
  MeanX = imcomplement(MeanX);
 %img = reshape(uint8(MeanX), fdim);
 %img=rot90(img);
 %img=rot90(img);
 %img=rot90(img);
 %img=fliplr(img);
 %imshow(img);
 %pause

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for c=1:cycle   
    for j=1:set      
         for p =1:repeat
            %pick a random number
            %r = round(rand_min + (rand_max-rand_min) .* rand(1,1));

            %New random input
            X  = reshape(double(XI(p,1,:)),[],1);
        
            %Desire output
            rX = reshape(double(XI(p,1,:)),[],1);
        
            %Center
            X = X-double(MeanX);
            nX = uint8(X);
            minimum = min(X);  %get minimum
            maximum = max(X);  %get maximum
            for z=1:m
                %Normalize pixel from 0-255
                nX(z,1) = 255*(X(z) -minimum)/(maximum-minimum);
            end
        
            X=double(nX);
            %Spiral image
            X = spiral_Image(X);
            rX= spiral_Image(rX);
            for q=1:train       
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %PCA
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
                %Update law    
                Y_PCA = W_PCA * X;
                dW = (eta_PCA * (( W_PCA * (rX) *(rX')) - (tril(Y_PCA*Y_PCA')*W_PCA)));
%                  W_PCA = ((set-1)/(set) * W_PCA )...
%                           + ((1/(set))*dW);
                       %W_PCA = double(uint8(((cT-1)/(cT) * W_PCA ) + ((1/(cT))*dW)));
                       %W_PCA = double(uint8(((set+repeat-1)/(set+repeat) * W_PCA ) + ((1/(set+repeat))*dW)));
                       W_PCA =((set+repeat-1)/(set+repeat) * W_PCA ) + ((1/(set+repeat))*dW);
            end
         end
    end
end
save (filename, 'W_PCA');