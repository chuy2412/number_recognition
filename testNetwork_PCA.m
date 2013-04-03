function testNetwork_PCA(W_PCA,set)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test Phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
counter =1;
r_max = set;
m = 28*28;
pixels =m;
fdim = [28,28];        %Dimentsion to show the image
parameters = 2;

%Range of random values
rand_min=1;
rand_max=10; 

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

 
for i =1:rand_max %For each digit
    
    %Get input
   
    ra = round(rand_min + (r_max-rand_min) .* rand(1,1));
    X  = reshape(double(XI(i,1,:)),[],1);
     %Display the result
    img = reshape(uint8(X), fdim);
    
    %Invert the image
    img = imcomplement(img);
    img=rot90(img);
    img=rot90(img);
    img=rot90(img);
    img=fliplr(img);
    subplot(rand_max, parameters,counter);
    
    strnum = int2str(i);

    imshow(img);
    counter=counter+1;
    %Paused to revise the result: 
    
     %Center
      X = X-double(MeanX);
     
     %Normalize (0-255)
     nX = uint8(X);
     minimum = min(X);  %get minimum
     maximum = max(X);  %get maximum
     for z=1:m
        %Normalize pixel from 0-255
        nX(z,1) = 255*(X(z) -minimum)/(maximum-minimum);
     end    
     X=double(nX);
        
     Y_PCA = W_PCA * X;
     Result = (W_PCA' * Y_PCA );% (W' *Y);    
    
    %Scale the result from 0-255 per pixel to display the result
    r = uint8(Result);
    minimum = min(Result);  %get minimum
    maximum = max(Result);  %get maximum
    for c=1:m
        %Normalize pixel from 0-255
        r(c,1) = 255*(Result(c,1) -minimum)/(maximum-minimum);
    end
    
    %Display the result
    subplot(rand_max, parameters,counter);
    img = reshape(r, fdim);
    img=rot90(img);
    img=rot90(img);
    img=rot90(img);
    img=fliplr(img);
    imshow(img);
    title(['Output PCA'  strnum]);
    counter= counter+1;
end