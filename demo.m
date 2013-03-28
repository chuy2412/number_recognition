clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Demonstration for number recognition using PCA and ICA
%Author: Jesus Rivera
%Values that can be edited to improve performance of ICA and PCA
%Please allow a couple of minutes to compute the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fdim = [28,28];        %Dimentsion to show the image
m = 28*28;             %m dimension
l = 160;                  %l the number of principal components
set=4200;
 W_ICA = 0.01*rand(l,m);
 W_PCA = 0.01*rand(l,m);
%W = 0.0001/m*rand(l,m);
Y = double(zeros(l));    
%Range of random values
rand_min=1;
rand_max=10; 
train =3;
 % eta = 0.0065/(m); 
 eta_ICA = 0.00000000000003;
 eta_PCA = 0.0065/(m); 
 %   eta = (0.00001/(m))/((set/rand_max)); 

error=1e-2;
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
newMean = mean((M0 + M1 + M2 + M3 + M4 + M5 + M6 +M7 +M8 + M9),2);
%newMean = mean((M1 +M2),2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Scale the mean so that the values are from 0-255
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xmean = newMean;
MeanX = uint8(Xmean);
minimum = min(Xmean);  %get minimum
maximum = max(Xmean);  %get maximum
for i=1:m
    %Normalize pixel from 0-255
    MeanX(i,1) = 255*(Xmean(i) -minimum)/(maximum-minimum);
end

% img = reshape(uint8(MeanX), fdim);
% img=rot90(img);
% img=rot90(img);
% img=rot90(img);
% img=fliplr(img);
% %imshow(img);
% pause(0.5)

% [comp ignore] = size(y);
I = eye(l);
cnt =1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:set      
     %for p =1:rand_max
        r = round(rand_min + (rand_max-rand_min) .* rand(1,1));
%         if(j==1)
%             r
%         end
%         r=p;
        %Train the network
        X  = reshape(double(XI(r,1,:)),[],1);
        rX = reshape(double(XI(r,1,:)),[],1);
            
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
        size(W_ICA);
        for q=1:train
            
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %ICA
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Demixing X with W
                y_ICA=W_ICA*X;
                
                %Activation function 4 from the supplemental notes
                %f=(3/4)*y.^11+(25/4)*y.^9+(-14/3)*y.^7+(-47/4)*y.^5+(29/4)*y.^3;
                f = y_ICA.^3;
                
                dW=(I-f*y_ICA')*W_ICA * eta_ICA;
%                 ndW = norm(dW);
                %Update the weight
                 W_ICA = W_ICA + dW;

                % Break if Algorithm diverges
                if (sum(sum(isnan(W_ICA)))>0) flag=1; 
                    break;
                end
                max(max(abs(dW)));

%                 % Break if Algorithm converges-- max weight update is less than error
%                 if (max(max(abs(dW)))<error & i>10) break; end; 
            
            
            
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %PCA
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
               %Update law    
               Y_PCA = W_PCA * X;
               dW = eta_PCA *...
                    (( W_PCA * (rX) *(rX')) - (tril(Y_PCA*Y_PCA')*W_PCA));
               W_PCA = ((set -1)/set * W_PCA ) + ((1/set)*dW);

            
%             max(max(W));
%              norm(W)


        end
         
     %end
end
   r

%[Y_ICA W_ICA] = ICA(tempX,W,set,train, eta);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test Phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
counter =1;
r_max = set;
for i =1:rand_max %For each digit
    
    %Get input
   
    ra = round(rand_min + (r_max-rand_min) .* rand(1,1));
    X  = reshape(double(XI(i,1,:)),[],1);
     %Display the result
    img = reshape(uint8(X), fdim);
    img=rot90(img);
    img=rot90(img);
    img=rot90(img);
    img=fliplr(img);
    subplot(rand_max, 3,counter);
    
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
     Result = (W_PCA'*Y_PCA );% (W' *Y);

    
    
    %Scale the result from 0-255 per pixel to display the result
    r = uint8(Result);
    minimum = min(Result);  %get minimum
    maximum = max(Result);  %get maximum
    for c=1:m
        %Normalize pixel from 0-255
        r(c,1) = 255*(Result(c,1) -minimum)/(maximum-minimum);
    end
    
    %Display the result
    subplot(rand_max, 3,counter);
    img = reshape(r, fdim);
    img=rot90(img);
    img=rot90(img);
    img=rot90(img);
    img=fliplr(img);
    imshow(img);
    title(['Output PCA'  strnum]);
    counter= counter+1;
    
     Y_ICA = W_ICA * X;
     Result = (W_ICA'*Y_ICA );% (W' *Y);

    %Scale the result from 0-255 per pixel to display the result
    r = uint8(Result);
    minimum = min(Result);  %get minimum
    maximum = max(Result);  %get maximum
    for c=1:m
        %Normalize pixel from 0-255
        r(c,1) = 255*(Result(c,1) -minimum)/(maximum-minimum);
    end
    
    %Display the result
    subplot(rand_max, 3,counter);
    img = reshape(r, fdim);
    img=rot90(img);
    img=rot90(img);
    img=rot90(img);
    img=fliplr(img);
    imshow(img);
    title(['Output ICA'  strnum]);
    counter= counter+1;
    
    
end