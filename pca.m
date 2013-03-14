clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Values that can be edited to improve performance of the adaptive PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fdim = [28,28];        %Dimentsion to show the image
m = 28*28;             %m dimension
l = 15;                  %l the number of principal components
train=10;                 %The number of epochs to train the weights W
W = 0.01/m*rand(l,m);       %Randomly initialize the Weights
eta = 0.0001/m;              %Learning rate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load data
I = load('mnist_all.mat');
%Variables
set=100;
totalImages = set;%*10;
dimension = [28,28];
pixels = m;
dim  = [totalImages,pixels];
newdim = [1,pixels];
Y = double(ones(l,1));          %Y matrix

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

%Get all the inputs
nMean1 = reshape(mean(XI(1,1:set,:),2),[],1);
nMean2 = reshape(mean(XI(2,1:set,:),2),[],1);

newMean = mean((nMean1 + nMean2),2);
t=1;
a=1;
b=2; 
for j=1:set      
     for p =1:2
        r = round(a + (b-a) .* rand(1,1));
        %Train the network
        X  = reshape(double(XI(r,j,:)),[],1);
        rX = reshape(double(XI(r,1,:)),[],1);
            
        %Center
        X = X-newMean;
        nX = uint8(X);
        minimum = min(X);  %get minimum
        maximum = max(X);  %get maximum
        for z=1:m
            %Normalize pixel from 0-255
            nX(z,1) = 255*(X(z) -minimum)/(maximum-minimum);
        end
        
        X=double(nX);
        
        %Update law    
        Y = W * X;
        W = eta * (( W * (rX) *(rX')) - (tril(Y*Y')*W));
            
%         max(max(W))
     end
end
r
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Scale the mean so that the values are from 0-255
%Functionality has been tested and it works fine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xmean = newMean;
MeanX = uint8(Xmean);
minimum = min(Xmean);  %get minimum
maximum = max(Xmean);  %get maximum
for i=1:m
    %Normalize pixel from 0-255
    MeanX(i,1) = 255*(Xmean(i) -minimum)/(maximum-minimum);
end

img = reshape(uint8(MeanX), fdim);
img=rot90(img);
img=rot90(img);
img=rot90(img);
img=fliplr(img);
imshow(img);
pause (2.5)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Show the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%newX = (W'*Y);
%min(newMean)
%max(newMean)
for i =1:1 %For each image
%for i=1:l
    %Calculate the result
    X  = reshape(double(XI(6,10,:)),[],1);
    newY = W * X;
     Result = W(:,:)'*Y;%+ double(MeanX);
    %Result = W(:,:)'*newY;%+ double(MeanX);
    
    
    %Scale the result from 0-255 per pixel to display the result
    r = uint8(Result);
    minimum = min(Result);  %get minimum
    maximum = max(Result);  %get maximum
    for c=1:m
        %Normalize pixel from 0-255
        r(c,1) = 255*(Result(c,1) -minimum)/(maximum-minimum);
    end
    
%     Result = double(r) + double(MeanX);
%     
%     r = uint8(Result);
%     minimum = min(Result);  %get minimum
%     maximum = max(Result);  %get maximum
%     for c=1:m
%         %Normalize pixel from 0-255
%         r(c,1) = 255*(Result(c,1) -minimum)/(maximum-minimum);
%     end
%     
    
    
    %Display the result
    img = reshape(r, fdim);
    img=rot90(img);
    img=rot90(img);
    img=rot90(img);
    img=fliplr(img);
    imshow(img);
    
     pause%Paused to revise the result: 
end
