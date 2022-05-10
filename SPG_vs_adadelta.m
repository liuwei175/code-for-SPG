%SPG v.s. adadelta for training the sparse autoencoder with tied weight.
%code for the paper: 
% LINEARLY-CONSTRAINED NONSMOOTH OPTIMIZATION FOR TRAINING AUTOENCODERS.
% to appear in SIAM Journal on Optimization

visibleSize=5;      % dimension of input data
hiddenSize =10;     % number of hidden units 
datasize=1;

fprintf('visibleSize=%d,  hiddenSize=%d, gaussian distribution. \n',visibleSize,hiddenSize);
fprintf(' SPG & SPG & SPG & SPG & Adadelta & Adadelta & Adadelta\n');
fprintf(' TrainErr & TestErr & FeaErr & Time  & TrainErr & TestErr  & Time & SPG succssess time & Adadelta succssess time\n')

maxiter=1001;   % max iteration for SPG
mm=1000;  %training dataset
mm1=300;  %test dataset

noise = 0.05;             % noise in data
lb1 = 0.1;                % weight decay parameter       
lb2 = 0.0001;             % weight of sparsity penalty term    


tt=0;loss=0;feasi=0;failtime=0;Adadeltatt=0;objada=0; testerror=0;failtimeada=0;testerrorada=0;terror=0;
for ss=1:1   %repeat ss times
    
   %-----------------------
   % generate the data
   testdata=0.5+mvnrnd(randn(visibleSize,1)',eye(visibleSize),mm+mm1); % random data under gaussian distribution
   testdata=testdata'+noise*randn(visibleSize,mm+mm1);
   testdata(testdata<0)=0;
   testdata2=testdata(:,(mm+1):end);testdata=testdata(:,1:mm);
   %-----------------------
  
    %initialization
    W1 = randn(hiddenSize, visibleSize)/mm;
    b1 = ones(hiddenSize, 1) * 0.1;
    V  = max(W1*testdata+repmat(b1,1,mm),0);
    b2=ones(visibleSize, 1) * 0.1;
    theta = [W1(:) ; b1(:); b2(:)]; 
    
    %-----------------------
    %% Perform optimisation -- Adadelta
    objFun = @(p) feedtied(p, hiddenSize, visibleSize, testdata, lb1,lb2);
    % Define the full gradient and the stochastic gradient functions
    gradStoch = @(i, x) gradcost(i, x, visibleSize, hiddenSize, lb1,lb2, testdata);
    nIter = 10000;  % max iteration for adadelta
    idxSG = randi(mm, 10, nIter);  %batch size:10.
    avgSG = @(idx, x) AvgGrad(gradStoch, idx, x);
    tic;
    
    %xMat_Adadelta denotes the output derived by Adadelta
    xMat_Adadelta = Adadelta(avgSG, theta, nIter, idxSG, 0.95);
    thetak=xMat_Adadelta(:,maxiter);
    % thetak=xMat_Adadelta(:,2001);
    Adadeltatt=Adadeltatt+toc;
    avgSG = @(idx, x) AvgGrad(gradStoch, idx, x);
    
    % compute trainerr, testerror by xMat_Adadelta
    for jj=1:size(xMat_Adadelta,2)
        trainerradadelta(jj) = feedtied(xMat_Adadelta(:,jj), hiddenSize, visibleSize, testdata,0);
        testerradadelta(jj)  = feedtied(xMat_Adadelta(:,jj), hiddenSize, visibleSize, testdata2,0);
    end
    if feedtied(xMat_Adadelta(:,end), hiddenSize, visibleSize, testdata,0)<1
        objada=objada+feedtied(xMat_Adadelta(:,end), hiddenSize, visibleSize, testdata,0);
        failtimeada=failtimeada+1;
        testerrorada = testerrorada+feedtied(xMat_Adadelta(:,end), hiddenSize, visibleSize, testdata2,0);
    end
    
    %-----------------------
    xMat_Adadelta=[];
    lossadmml1=[];feal1=[];lossadmm1=[];testerror=[];
    
    %-----------------------
    
    %-----------------------
    % run SPG
    spg_main;
    if min(lossadmm1(2:end))<1
        failtime=failtime+1;
        tt = tt+tttl1;
        loss= loss+min(lossadmm1(2:end));
        terror= terror+min(testerror);
        feasi=feasi+min(feal1);
    end
end

%-----------------------
%plot the figure
fprintf(' %.3e & %.3e & %.3e& %.3f & %.3e & %.3e& %.3f & %d &%d\\\\\n',...
        loss/failtime,terror/failtime,feasi/failtime, tt/failtime,objada/failtime,testerrorada/failtime,...
        Adadeltatt/failtime,failtime,failtimeada);
lossadmml1=[];feal1=[];
fprintf('\\hline\n');



colors = [0 0 0; 255 0 0; 0 0 255; 255 0 255; 192 192 192; 128 128 105; 244 164 96; 8 46 84; 
    210 105 30; 61 89 171; 0 210 87; 0 255 127; 65 105 225;221 160 221; 227 23 13]/255;
lines   = {'k-' 'b--' 'r-.' 'c:' 'g-*' 'k-d'};


time1=0:Adadeltatt/(size(trainerradadelta,2)-1):Adadeltatt;
time2=0:tttl1/(size(lossadmm1,2)-1):tttl1;
idx1 = 1: (nIter)/50: (nIter+1);
figure
semilogy(time1(idx1),trainerradadelta(idx1),lines{1},'Color',colors(1,:));hold on
semilogy(time1(idx1),testerradadelta(idx1),lines{2},'Color',colors(2,:));hold on
semilogy((time2)+Adadeltatt/10, (lossadmm1),lines{3},'Color',colors(3,:));hold on
semilogy((time2)+Adadeltatt/10, (testerror),lines{4},'Color',colors(4,:));hold on
%% Plot results -- Convergence plot -- Purely stochastic gradient
hold off
grid on
xlabel('Time (s)');
ylabel('TrainErr/TestErr');%ylim([min(loss3),max(sgdobj)*8]);
solvers = { ...
    'Adadelta (TrainErr)','Adadelta (TestErr)','SPG (TrainErr)', 'SPG (TestErr)'
    };
legend(solvers,'location','best');
set(gca,'linewidth',1.6);
set(gca,'fontsize',16,'fontweight','bold');
%-----------------------


function xMat = Adadelta(sg, x0, nIter, idxSG, beta, epsilon)
    %ADADELTA Adadelta algorithm for SGD optimisation (Zeiler, 2012)
    %
    % Implemented according to preprint 1212.5701v1, 22 Dec 2012.
    %
    % This function minimises an objective function `J(x)`, where `x` is an
    % n-dimensional column vector containing decision variables. The stochastic
    % gradient of the objective is supplied as the function handle `sg`, which
    % accepts the index (or indices) of the stochastic gradient as the first
    % argument and the value of the decision variable as the second argument,
    % i.e. `sg(idx, x)`. `sg` returns an n-dimensional column vector.
    %
    % Note that both `idx` and `x` must be column vectors. If `idx` is a
    % vector, function `sg` should return the averaged stochastic gradient. You
    % can use the `AvgGrad` wrapper provided in this repo to do the averaging
    % without additional effort.
    %
    % `idxSG` is a row vector or a matrix which columns specify the indices of
    % the stochastic gradient should be used at each iteration. If `idxSG` has
    % fewer columns than `nIter`, it is repeating to the required size.
    %
    % Normally, one would generate `idxSG` with `randi`, e.g. `idxSG =
    % randi(<maxIdx>, 1, nIter);`.
    %
    % Refer to [1] for a description of solver parameters `delta` and
    % `epsilon`.
    %
    % References:
    %   [1] Zeiler, Matthew D. Adadelta: An Adaptive Learning Rate Method.
    %   arXiv preprint: http://arxiv.org/abs/1212.5701
    %
    % Input:
    %   sg       : function handle to the stochastic gradient
    %   x0       : initial guess for the decision variables
    %   nIter    : number of iterations to perform
    %   idxSG    : indices of the gradients to use
    %   beta     : exponential decay rate for moving averages
    %   epsilon  : back-to-numerical-reality addend, default: `sqrt(eps)`
    %
    % Output:
    %   xMat     : matrix with decision variables at each iteration step
    %

    % Store default value for `epsilon` if there are only 5 input arguments
    if nargin == 5
        epsilon = 1.0e-6;
    end

    % Store the number of decision variables
    nDecVar = length(x0);

    % Allocate output
    xMat = zeros(nDecVar, nIter + 1);

    % Set the initial guess
    xMat(:, 1) = x0;

    % Repeat `idxSG` if it has fewer columns than `nIter`
    if size(idxSG, 2) < nIter
        idxSG = repmat(idxSG, 1, ceil(nIter/size(idxSG, 2)));
        idxSG(:, nIter + 1 : 1 : end) = [];
    end

    % Initialise accumulator variables
    accG = zeros(nDecVar, 1); % accumulated gradients
    accD = zeros(nDecVar, 1); % accumulated updates (deltas)

    % Run optimisation
    for i = 1 : 1 : nIter
        % Get gradients w.r.t. stochastic objective at the current iteration
        sgCurr = sg(idxSG(:, i), xMat(:, i));

        % Update accumulated gradients
        accG = beta.*accG + (1 - beta).*(sgCurr.^2);

        % Compute update
        dCurr = -(sqrt(accD + epsilon)./sqrt(accG + epsilon)).*sgCurr;

        % Update accumulated updates (deltas)
        accD = beta.*accD + (1 - beta).*(dCurr.^2);

        % Update decision variables
        xMat(:, i + 1) = xMat(:, i) + dCurr;
    end

end

function grad = gradcost(i, theta, visibleSize, hiddenSize, lb1,lb2, data)

    % compute the gradient of the network

    % visibleSize: the number of input units (probably 64) 
    % hiddenSize: the number of hidden units (probably 25) 
    % lambda: weight decay parameter
    % sparsityParam: The desired average activation for the hidden units (denoted in the lect4ure
    %                           notes by the greek alphabet rho, which looks like a lower-case "p").
    % beta: weight of sparsity penalty term
    % data: Our 64xmm matrix containing the training data.  So, data(:,i) is the i-th training example. 

    % The input theta is a vector (because minFunc expects the parameters to be a vector). 
    % We first convert theta to the (W1, W2, b1, b2) matrix[表情]ector format, so that this 
    % follows the notation convention of the lecture notes. 

    W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    W2=  W1';
    b1 = theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);
    b2 = theta(hiddenSize*visibleSize+hiddenSize+1:end);

    cost = 0;
    W1grad = zeros(size(W1)); 
    W2grad = zeros(visibleSize,hiddenSize); % we represent W2 in another way
    b1grad = zeros(size(b1)); 
    b2grad = zeros(size(b2));

    n=size(data,2);

    mm=1; x=data(:,i);
    z2=(W1*x)+b1;
    a2=max(z2,0);
    z3=(W2*a2)+b2;
    a3=max(z3,0);

    %cost=1/2[表情]*sum(sum((a3-data).^2))+lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));


     reluz2=relugradient(z2);

     delta3=-(x-a3).*relugradient(z3);
     delta2=W2'*delta3.*reluz2;
     W1grad=W1grad+delta2*x';
     W2grad=W2grad+delta3*a2';
     b1grad=b1grad+sum(delta2,2);
     b2grad=b2grad+sum(delta3,2);

    b1grad=b1grad/mm+lb2*reluz2;
    b2grad=b2grad/mm;
    W1grad=W1grad/mm+lb1*W1+lb2*reluz2*x';
    W2grad=W2grad/mm;
    W1grad=W1grad+W2grad';

    %-------------------------------------------------------------------
    % After computing the cost and gradient, we will convert the gradients back
    % to a vector format (suitable for minFunc).  Specifically, we will unroll
    % your gradient matrices into a vector.

    grad = [W1grad(:) ; b1grad(:) ; b2grad(:)];

end

function cost = feedtied(theta, hiddenSize, visibleSize, data,lb)

    % compute the function value of the network
    % theta: trained weights from the autoencoder
    % visibleSize: the number of input units (probably 64) 
    % hiddenSize: the number of hidden units (probably 25) 
    % data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

    % We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
    % follows the notation convention of the lecture notes. 

    W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    b1 = theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);
    W2 = W1';
    b2 = theta(hiddenSize*visibleSize+hiddenSize+1:end);

    %% ---------- YOUR CODE HERE --------------------------------------
    %  Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.
    activation  = RELU(W1*data+repmat(b1,[1,size(data,2)]));
    %construction = sigmoid(W2*activation+repmat(b2,[1,size(data,2)]));
    construction = RELU(W2*activation+repmat(b2,[1,size(data,2)]));

    cost=dist2(construction,data)/size(data,2)+lb*sum(sum(activation))+lb*sum(sum(W1.^2));
    %-------------------------------------------------------------------

    end

function avgSG = AvgGrad(sg, idx, x)
    %AVGGRAD Compute an average of stochastic gradients
    %
    % This function can be used to wrap a stochastic gradient function which
    % accepts only scalar stochastic gradient indices.
    %
    % Input:
    %   sg    : function handle to the stochastic gradient
    %   idx   : indices of the gradients to use, column vector
    %   x     : value of the decision parameters
    %
    % Output:
    %   avgSG : averaged stochastic gradient
    %

    avgSG = zeros(length(x), length(idx));

    for i = 1 : 1 : length(idx)
        avgSG(:, i) = sg(idx(i), x);
    end

    avgSG = mean(avgSG, 2);

end

%-------------------------------------------------------------------
% Here's an implementation of the relu function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 


function relu = RELU(x)
    relu = max(0,x);
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function kld = absgradient(x)
    kld=x;
    kld(kld<0)=-1;
    kld(kld>0)=1;
end 
function kld = relugradient(x)
    kld=x;
    kld(kld<0)=0;
    kld(kld>0)=1;
end 

function g = sigmoidGradient(z)
    g=zeros(size(z));
    g=sigmoid(z).*(1-sigmoid(z));
end

function n2 = dist2(x, c)
    %DIST2	Calculates squared distance between two sets of points.
    %
    %	Description
    %	D = DIST2(X, C) takes two matrices of vectors and calculates the
    %	squared Euclidean distance between them.  Both matrices must be of
    %	the same column dimension.  If X has M rows and N columns, and C has
    %	L rows and N columns, then the result has M rows and L columns.  The
    %	I, Jth entry is the  squared distance from the Ith row of X to the
    %	Jth row of C.
    %   D(i,j) = norm(X(i,:)-C(j,:)).^2;
    %
    %	See also
    %	GMMACTIV, KMEANS, RBFFWD
    %

    %	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

    n2=sum(sum((x-c).^2));

end
