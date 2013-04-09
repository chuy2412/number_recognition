%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backpropagation network for number recognition
% Phase2.m
% Author: Jesus Rivera
% References: *Suplemental Notes
%             *"Neural Networks and Learning Machines" 3rd Edition. CH4
%             *Stackoverflow
%             *Lec-19 Back Propagation Algorithm
%                  www.youtube.com/watch?v=nz3NYD73H6E
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load data
I = load('mnist_all.mat');
X1 = double(reshape(I.train1(1,:),[],1));
X2 = double(reshape(I.train2(1,:),[],1));
X3 = double(reshape(I.train3(1,:),[],1));
X4 = double(reshape(I.train4(1,:),[],1));
X5 = double(reshape(I.train5(1,:),[],1));
X6 = double(reshape(I.train6(1,:),[],1));
X7 = double(reshape(I.train7(1,:),[],1));
X8 = double(reshape(I.train8(1,:),[],1));
X9 = double(reshape(I.train9(1,:),[],1));
X0 = double(reshape(I.train0(1,:),[],1));


%Inputs
%X = [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9]
X = [X0 X1];


%Desired outputs
%DDJ = [0 1 2 3 4 5 6 7 8 9]
DDJ = [0 1];    

ETA = 10;
[nInputs N] = size(X);
[nOutputs N] = size(DDJ);
nHidden = 784; %Two hidden units



%Initialize the weights with small random values
wji = 0.075*rand(nHidden,nInputs+1);
wkj = 0.075*rand(nOutputs,nHidden+1);
netj = zeros(1,nHidden);
netk = zeros(1,nOutputs);
Oj = zeros(1,nHidden);
Ok = zeros(1,nOutputs);
deltaK = zeros(1,nOutputs);
deltaJ = zeros(1,nOutputs);
ek = zeros(1,nOutputs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training Phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iterations = 0; %Number of iterations
Error_Steps=2;
%Stopping Criteria: 10000 iterations or The error of the last 4 steps is 
%Less than 4%
fprintf('Starting Training Phase...\n');
while((norm(Error_Steps)> 0.04) && (iterations < 100))
    Error_Steps=0; 
    for i = 1:N  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Forward pass 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = 1:nHidden
            netj(j) = wji(j,1:end-1)*(X(:,i))+ wji(j,end);
            %Sigmoidal function for j
            Oj(j) = 2 / (1+exp(-netj(j))) - 1 ;
        end
        %hidden to output layer
        eSum=0;
        for k = 1:nOutputs         
           netk(k) = wkj(k,1:end-1)*Oj' + wkj(k,end);
           %Sigmoidal function for k
           Ok(k) = 2/(1+exp(-netk(k))) -1;
           ek(k) = DDJ(k,i) - Ok(k); 
           %derivative of Sigmoidal of netk
           derivativeSk =  (1+ Ok(k))*(1-Ok(k)) / 2 ;
           deltaK(k) =derivativeSk*ek(k);
           eSum = eSum+ (ek(k))^2;
        end
        
        Error_Steps = Error_Steps + abs(ek);
       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Backpropagation Phase
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = 1:nHidden
            sum=0;  
            for k = 1:nOutputs
               %Summation for wk weights
               sum = sum + wkj(k,j)*deltaK(k);
            end
            %Derivative of Sigmoidal of netj
            derivativeSj = (1+ Oj(j))*(1-Oj(j)) / 2;
            deltaJ(j) =derivativeSj * sum;
           % deltaJ(j) = (1-Oj(j)) *sum;
        end 
        
        %Update weights hidden to output
        for k = 1:nOutputs
            for l = 1:nHidden
                wkj(k,l) = wkj(k,l)+ ETA*deltaK(k)*Oj(l);
            end
            wkj(k,l+1) = wkj(k,l+1)+ ETA*deltaK(k);
        end  
        
        %Update weights input-hidden
        for j = 1:nHidden
            for p = 1:nInputs
                wji(j,p) = wji(j,p) + ETA *deltaJ(j)*X(p,i);
            end
            wji(j,p+1) = wji(j,p+1) + ETA *deltaJ(j);
         end    
    end
     iterations = iterations+1 ;
end
fprintf('It took %d iterations \n',iterations);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test inputs 
%test =[X1 X3 X5 X7 X9 X2 X4 X6 X8 X0];
 test = [X1 X0];
 
%Desired outputs
%d = [1 3 5 7 9 2 4 6 8 0];
d = [1 0 ];             %desired outputs

fprintf('Starting Testing Phase...\n');
for i = 1:2 
    for j = 1:nHidden
        netj(j) = wji(j,1:end-1)*(test(:,i))+ wji(j,end);
        %Sigmoidal function for j
        Oj(j) = 2 / (1+exp(-netj(j))) - 1 ;
    end
    %hidden to output layer
    eSum=0;
    for k = 1:nOutputs         
        netk(k) = wkj(k,1:end-1)*Oj' + wkj(k,end);
        %Sigmoidal function for k
        Ok(k) = 2/(1+exp(-netk(k))) -1;
        ek(k) = d(k,i) - Ok(k); 
        %derivative of Sigmoidal of netk
        derivativeSk =  (1+ Ok(k))*(1-Ok(k)) / 2 ;
        deltaK(k) =derivativeSk*ek(k);
        eSum = eSum+ (ek(k))^2;
    end
        
   e = d(i) - Ok;
   fprintf('%f  (error =%f) \n',Ok,e);  
end



