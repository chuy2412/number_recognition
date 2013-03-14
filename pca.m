clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Values that can be edited to improve performance of the adaptive PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fdim = [28,28];        %Dimentsion to show the image
m = 28*28;             %m dimension
l = 15;                  %l the number of principal components
train=20;                 %The number of epochs to train the weights W
W = 0.1/m*rand(l,m);       %Randomly initialize the Weights
eta = 0.00001/m;              %Learning rate
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
Y = double(ones(l,10));          %Y matrix

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
%X = double(zeros(pixels,totalImages));
%Xmean = double(X);

%Get all the inputs
t=1;
 for i=1:1
     for j=1:set
         X  = reshape(double(XI(i,j,:)),[],1);
         
         %Calculate the mean
         if t==1
             Xmean = X;
         else
             Xmean = ((t-1)/t)*Xmean + (1/t)*double(X);
         end 
         t=t+1; %update the counter
         %Center
            X = X-Xmean;
            nX = uint8(X);
            minimum = min(X);  %get minimum
            maximum = max(X);  %get maximum
            for z=1:m
                %Normalize pixel from 0-255
                nX(z,1) = 255*(X(z) -minimum)/(maximum-minimum);
            end
            X=double(nX);
         
         for p =1:train
            %Train the network
            Y(:,i) = W * X; %Original
            
            W = eta * (( W * (X) *(X')) - (tril(Y(:,i)*Y(:,i)')*W));%Original
            
            max(max(W))
         end
     end
 end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Scale the mean so that the values are from 0-255
%Functionality has been tested and it works fine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MeanX = uint8(Xmean);
minimum = min(Xmean);  %get minimum
maximum = max(Xmean);  %get maximum
for i=1:m
    %Normalize pixel from 0-255
    MeanX(i,1) = 255*(Xmean(i) -minimum)/(maximum-minimum);
end

img = reshape(uint8(MeanX), fdim);
%img = reshape(uint8(otherMean), fdim);
img=rot90(img);
img=rot90(img);
img=rot90(img);
img=fliplr(img);
imshow(img);
pause (2.5)

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Center the data
% %Functionality has been tested and it works fine
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xcentered = X;
% for i=1:totalImages
%      Xcentered(:,i) =X(:,i) - double(MeanX); 
%      X(:,i) = Xcentered(:,i);
%      
%      %Xcentered(:,i) = Xcentered(:,i); 
%      
%      
%    %Remove comments to see the new centered values
%      img = reshape(uint8(X(:,i)), fdim);
%      img=rot90(img);
%      %imshow(img);
%      %pause (0.6)
% end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Train the weights
% %Using Lower triangle
% %Based on Suplemental Notes from PCA page 18
% %Not sure why is not working... At some point, the Y becomes NaN
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a=1;
% b=totalImages;
% for i =1:train
%     %for j=1:totalImages
%         Y = W*X; %Original
%         %j = round(a + (b-a) .* rand(1,1));
%         %Y=W*X(:,j);
%         
%         %Y(:,j) = W * X(:,j); 
%         %Form a)
%         %W = eta * (( Y * X') - (tril(Y*Y')*W));
%     
%         %Form b)
%         W = eta * (( W * (X) *(X')) - (tril(Y*Y')*W));%Original
%        % W = eta * (( W * (X(:,j)) *(X(:,j)')) - (tril(Y*Y')*W));
%     %end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Show the results
%Does not work well
%The Y matrix has multiple values as NaN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Currently only shows one result per image by using l principal
% components. By default, l is set to 10 just to verify functionality
%This can be edited to show multiple principal components per image
%once the function works well (current problem is the update law)
% The matrix Y contains multiple NaN values which produces a NaN
%result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%newX = (W'*Y);
%min(newMean)
%max(newMean)
for i =1:10 %For each image
%for i=1:l
    %Calculate the result
    Result = W(:,:)'*Y(:,i);% + double(MeanX);
    
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
    
    pause (0.5) %Paused to revise the result: 
end
