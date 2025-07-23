clear;close all;clc
addpath('./functions');
%% load network
load Untrained_5blocks_10801080_16cm_sin_sin_ws.mat
%% dataset
% load dataset
rawImagePath = 'DIV2K';
imds = imageDatastore(rawImagePath,'IncludeSubfolders',true);
augimds = augmentedImageDatastore([1080 1080],imds,'ColorPreprocessing',"rgb2gray");
% initialize plot
[ax1,ax2,lineLossRec1,lineLossRec2,lineLoss]=initializePlots4();
plotFrequency = 5;
%% training parameters
numEpochs = 10;  
miniBatchSize = 1;  
augimds.MiniBatchSize = miniBatchSize;
averageGrad = [];
averageSqGrad = [];
numIterations = floor(augimds.NumObservations*numEpochs/miniBatchSize)*10;
mbq = minibatchqueue(augimds,'MiniBatchSize',miniBatchSize,'MiniBatchFormat','SSBC');
learnRate = 4e-3;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
executionEnvironment = "auto";
%% training
iteration = 0;
pic_num = 1;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        
        % Read mini-batch of data.
        [dlX] = next(mbq);
  
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end

        for k=1:10
         iteration = iteration + 1;
         
        % Evaluate model gradients. 
         [gradients,dlY,lossRec1,lossRec2,loss] = dlfeval(@modelGradients,dlnet,dlX);

        % Update the network parameters using the Adam optimizer.
        [dlnet,averageGrad,averageSqGrad] = ...
            adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
    
        addpoints(lineLossRec1,iteration,double(gather(extractdata(lossRec1))))
        addpoints(lineLossRec2,iteration,double(gather(extractdata(lossRec2))))
        addpoints(lineLoss,iteration,double(gather(extractdata(loss))))
  
        % Every plotFequency iterations, plot the training progress.
        if mod(iteration,plotFrequency) == 0            
            % Use the first image of the mini-batch as a validation image.
            dlV = dlX(:,:,:,1);
            % Use the transformed validation image computed previously.
            dlVY = dlY(:,:,:,1);
            dlVY = rescale(dlVY,0,255);
            dlZ = forward(dlnet,dlX,'Outputs','sinpi');
            dlVZ = dlZ(:,:,:,1);
            dlVZ = rescale(dlVZ,0,255);
            
            % To use the function imshow, convert to uint8.
            validationImage = mat2gray(uint8(gather(extractdata(dlV))));
            transformedValidationImage = mat2gray(uint8(gather(extractdata(dlVY))));
            phaseImage = mat2gray(uint8(gather(extractdata(dlVZ))));
            
            % Plot the input image and the output image and increase size
            imshow(imtile({validationImage,transformedValidationImage,phaseImage},'GridSize', [1 3]),'Parent',ax2);
        end
        
        % Display time elapsed since start of training and training completion percentage.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        completionPercentage = round(iteration/numIterations*100,2);
        title(ax1,"Epoch: " + epoch + ", Iteration: " + iteration +" of "+ numIterations + "(" + completionPercentage + "%)"+...
            ", LearnRate: "+ learnRate + ", Elapsed: " + string(D))
        drawnow
        
       end
      
    end
    
    learnRate=learnRate*0.9;

end

save('Trained_5blocks_10801080_16cm_sin_sin_ws.mat','dlnet','averageGrad','averageSqGrad');
%% loss function
function [gradients,dlY,lossRec1,lossRec2,loss] = modelGradients(dlnet,dlX)

    [dlY] = forward(dlnet,dlX,'Outputs','crop');loss_0 = TotalVariationLoss(dlY);
   
    [dlY_1] = forward(dlnet,dlX,'Outputs','crop1');loss_1 = TotalVariationLoss(dlY_1);
  
    [dlY_2] = forward(dlnet,dlX,'Outputs','crop2'); loss_2 = TotalVariationLoss(dlY_2);

    [dlY_3] = forward(dlnet,dlX,'Outputs','crop3'); loss_3 = TotalVariationLoss(dlY_3);

    [dlY_4] = forward(dlnet,dlX,'Outputs','crop4'); loss_4 = TotalVariationLoss(dlY_4);
  
    X = gather(extractdata(dlX));X1 = mat2gray(X); dlX1 = dlarray(X1, 'SSCB');
    loss_5 = mseLoss((dlY+0.88*dlY_1+0.88*dlY_2+0.45*dlY_3+0.45*dlY_4).^6,dlX1.^6);

    % Calculate the total loss.
    loss = loss_0+loss_1+loss_2+loss_3+loss_4+loss_5;
    gradients = dlgradient(loss,dlnet.Learnables);
    
end
function loss = mseLoss(dlA,dlB)

    loss = mean((dlA-dlB).^2,'all');
    
end
function loss = npccLoss(dlA,dlB)

A = dlA - mean(dlA,[1 2]);
B = dlB - mean(dlB,[1 2]);
A_norm = sqrt(sum(A.^2,[1 2]));
B_norm = sqrt(sum(B.^2,[1 2]));
npcc = -sum(A.*B,[1 2])./(A_norm.*B_norm);
loss = mean(npcc,'all');

end
function tv_val = TotalVariationLoss(X)

dX1 = X(2:end, :, :, :) - X(1:end-1, :, :, :);
dX2 = X(:, 2:end, :, :) - X(:, 1:end-1, :, :);
[S1, S2, B, C] = size(X);
numDiffs = (S1-1)*S2*B*C + S1*(S2-1)*B*C;  
tv_val = (sum(abs(dX1(:))) + sum(abs(dX2(:)))) / numDiffs;

end
