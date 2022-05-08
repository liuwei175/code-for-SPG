% Recent revision by Liu Wei was on May 5th, 2020
% main code for SPG 
% one can delete % as follows and try other problems
%-------------------------------------
% clear;
% % 
% visibleSize=5;mm1=30;hiddenSize =5;mm=100;
% testdata=0.5+mvnrnd(randn(visibleSize,1)',eye(visibleSize),mm+mm1)';
% testdata=rand(visibleSize,mm+mm1);
%  testdata=testdata+0.05*randn(visibleSize,mm+mm1);
%  testdata(testdata<0)=0; %testdata(testdata>1)=1;
% %  %testdata=testdata/max(max(testdata));
%  testdata1=testdata(:,1:mm);testdata2=testdata(:,(mm+1):end);
%  testdata=testdata1;
% %  hiddenSize =10;     % number of hidden units 
% % % sparsityParam = 0.1;   % desired average activation of the hidden units.
% % %                         % here, we use KL distance, we may use some other
% % % %                         % sparsities, such as group sparsity, or some else.
%  lb1 = 0.1;                % weight decay parameter       
%  lb2 = 0.0001;                % weight of sparsity penalty term    
% % % % 
% % % 
% load('data3.mat')
% mm=10000;hiddenSize=2000;visibleSize=784;testdata=testdatachoose(1000);
% testdata=testdata/max(max(testdata));
%_------------
 beta= 1/mm;   %penalty parameter
 %_------------
% %  display a random sample of 200 patches from the dataset
% tiedweight_activation=5;  %0-sigmoid; 1-tied+sigmoid; 2-relu; 3-tied+relu; 4-relu+sigmoid; 5-relu+sigmoid+tied;
% %  Randomly initialize the parameters
% 
% 
% thetak = inismall(hiddenSize, visibleSize,testdata,1,mm);
% thetak=thetakk;
%-------------------------------------

W1 = reshape(thetak(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = thetak(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);
b2 = thetak(hiddenSize*visibleSize+hiddenSize+1:end);
V  = max(W1*testdata+repmat(b1,1,mm),0);
Lu=1;Lv=1;Ltt=1;

% initial smoothing parameter
mu(1)=0.001;

a1_test = max(W1*testdata+repmat(b1,1,mm),0);
a2_test = max(W1'*a1_test+repmat(b2,1,mm),0);
lossadmm1=[];lossadmml1=[];testerror=[];lossadmml2=[];costt=[];feal1=[];
lossadmm1(1) = dist2(a2_test,testdata)/mm;
lossadmml1(1) = lossadmm1(1)+lb1*sum(sum(a1_test))+lb2*sum(sum(W1.^2));
lossadmml2(1) = lb1*sum(sum(a1_test))+lb2*sum(sum(W1.^2));
a11_test = max(W1*testdata2+repmat(b1,1,mm1),0);
a22_test = max(W1'*a11_test+repmat(b2,1,mm1),0);
testerror(1) =dist2(a22_test,testdata2)/mm1;
tttl1=0;
aaa=1.1;
for i=1:3000
  
   
    theta=[W1(:);b1(:);b2(:)]; 
    
%     tic;

    % compute the gradient
    
    [cost,gradW,gradb1,gradb2,gradV,tttid] = tildeF(theta,V,mu(i),hiddenSize,beta,testdata);
    
     tttl1=tttl1+tttid;
     
     % compute the objective function value
    
    costt(i)=cost+1/mm*sum(sum(testdata.^2))+lb2*sum(sum(a1_test))+lb1*sum(sum(W1.^2));
    
    % update the smoothing parameter
    if i>1 && costt(i)-costt(i-1)>=-max(power(mu(i),2)/Lu,1e-16) && costt(i)-costt(i-1)<=1e-6
        mu(i+1)=mu(i)/2;Lu=Lu*aaa;Lv=Lv*aaa;
    else 
        mu(i+1)=mu(i);
    end
     
    for j=1:5
        
        % update W,b_1 by sub-algorithm tied_sub
        % ttadam: time
        % kktvio: kktviolation
        [W1t,b1t,Vt,ttadmm,kktvio(i)] = tied_sub(W1,b1,V,gradW,gradb1,gradV,testdata,Lu,Lu,Lu,Ltt,lb1,lb2);
        
        % update W,b_2
        b2t = b2 - gradb2/Lv;    
        
        tttl1=tttl1+ttadmm;
        
        thetat=[W1t(:);b1t(:);b2t(:)];
        
        %-----------------------
        % 
        a1_test = max(W1t*testdata+repmat(b1t,1,mm),0);
        a2_test = max(W1t'*a1_test+repmat(b2t,1,mm),0);
        a11_test = max(W1t*testdata2+repmat(b1t,1,mm1),0);
        a22_test = max(W1t'*a11_test+repmat(b2t,1,mm1),0);
        lossadmm1(i+1) =dist2(a2_test,testdata)/mm;             % trainerr
        testerror(i+1) =dist2(a22_test,testdata2)/mm1;          % testerr
        lossadmml1(i+1) = dist2(a2_test,testdata)/mm+lb2*sum(sum(a1_test))+lb1*sum(sum(W1.^2));
        lossadmml2(i+1) = lb2*sum(sum(a1_test))+lb1*sum(sum(W1.^2));         
        %-----------------------
        
        %adjust the stepsize
        if i>1 && j>1 && costt(i)-costt(i-1)>-0.0001*(thetat-theta)'*(thetat-theta)
            Lu=Lu*1.2;
            Lv=Lv*1.1;
            %Ltt=Ltt*1.01;
        else
          break;
        end
    end
%     kktviol1(i)=dist2(W1,W1t)+dist2(b1,b1t)+dist2(V,Vt);
%     kktviol1(i)=kktviol1(i)*Lu;
   
    %if i>1 && lossadmm1(i)>100
    %    break;
    %end
    
    % feasibility error
    feal1(i)=dist2(a1_test,Vt)/mm/hiddenSize;
          
    W1=W1t;b1=b1t;V=Vt;Lu1(i)=Lu;

    %stop criterion
    if mu(i+1)<=1e-6 || tttl1>Adadeltatt/ss%|| lossadmm1(i+1)<1e-3 || i==1000
       break;
    end

    
end

function [W,b1,V,ttadmm,kktvio] = tied_sub(Wk,b1k,Vk,gradW,gradb1,gradV,data,LW,Lb,Lv,L2,lb1,lb2)
    %[W,b1,V,costadmm] = tied_sub(Wk,b1k,Vk,gradW,gradb1,gradV,data,LW,Lb,Lv)
    %<gradW,W-Wk>+<gradb,b-bk>+<gradV,V-Vk>+L1/2|W-Wk|^2+L1/2|b-bk|^2+L1/2|V-Vk|^2
    %s.t.V>=WX+repmat(b,1,n)
    %    V>=0
    %alm
    %<gradW,W-Wk>+<gradb1,b1-b1k>+<gradV,V-Vk>+LW/2|W-Wk|^2+Lb/2|b-bk|^2+LV/2|V-Vk|^2+<gamma,U-WX-B1>+L2/2|U-WX-B1|^2
    %s.t.V>=U
    %    V>=0
    %indexdiv=0;
    if nargin < 12
        lb1=0;
        lb2=0;
    end
    [N1,n]=size(Vk);
    N0=size(Wk,2);
    gamma=zeros(N1,n);
    data=[data;ones(1,n)];
    Whatk=[Wk,b1k];
    gradWhat=[gradW,gradb1];
    gradVhat=gradV-Lv*Vk;
    U=Whatk*data;

    Lwb=[repmat(LW,1,N0),Lb];
    Lwb=diag(Lwb);
    
    Lwb2=[repmat(LW+lb1,1,N0),Lb];
    Lwb2=diag(Lwb2);

    WWden = Lwb2+L2*data*data';
    WWmol = Whatk*Lwb-gradWhat;
    invWW=inv(WWden);
    
    gradVhat=gradV-Lv*Vk+lb2;
    
    Lgamma=1.618*min(L2,1);
    
    ttadmm=0;maxii=40000;
    for i=1:maxii

        tic;

        What = (WWmol+gamma*data'+L2*U*data')*invWW;
        %cost2 = <gradW,W-Wk>+<gradb,b-bk>+<gradV,V-Vk>+L1/2|W-Wk|^2+L1/2|b-bk|^2+L1/2|V-Vk|^2

        Uhat=What*data;

        [V,U2] = updateUV_tied(Lv,L2,gradVhat,gamma-L2*Uhat);

        Uvar= U2-Uhat;
        
        gamma = gamma+Lgamma*Uvar;
        
        ttadmm=ttadmm+toc;
        
        testUvar=sum(sum(abs(Uvar.^2)));
        
        if testUvar<1e-6 || i==maxii
            sumU=sum(sum(abs((U-U2).^2)));
            sumU2=sum(sum(abs(((U-U2)*data').^2)));
            kktvio=testUvar*2+sum(sum(abs((Uvar*data').^2)))+2*sumU+sumU2;
            break;
        end  
        
        %if sum(sum(abs(Uvar.^2)))/n>1e6
        %   indexdiv=1;
        %    break
        %end
        
        U=U2;

    end
    W=What(:,1:N0);
    b1=What(:,end);
end


function [V,U] = updateUV_tied(a,b,c1,c2)
    %min a/2(x+c1/a)^2+b/2(y+c2/b)^2 
    %    s.t. x>=y, x>=0
    U=zeros(size(c1));
    V=zeros(size(c1));
    c1=c1/a;c2=c2/b;
    index=c1<=c2 & c1<=0;
    V(index)=-c1(index);
    U(index)=-c2(index);
    index=c2>=0 & c1>0;
    V(index)=0;
    U(index)=-c2(index);
    index=c1>c2 & c2<=0 & c1*a+c2*b<=0;
    V(index)=-(c1(index)*a+c2(index)*b)/(a+b);
    U(index)=V(index);
    index=c1>c2 & c2<=0 & c1*a+c2*b>0;
    V(index)=0;
    U(index)=0;
end

function [cost,gradW,gradb1,gradb2,gradV,tt] = tildeF(theta,V,mu,hiddenSize,beta,testdata)
    %tildeF= ﻿\tilde F(W,b,V)=1/n sum \|(W^Tv_i+b_2)_{+}\|_2^2+1/n |X|_F^2+ lb1 sum e^Tv_i+lb2 \|W\|_F^2...
    ...-sum x_i^T\tilde{f} (W^Tv_i+b_2,\mu)/n*2+\beta\sum_{i=1}^{n}e^T\left(v_i-\tilde{f}(Wx_i+b_{1},\mu).
    % caculate the function value and gradient of the smoothing function of F
    % bbeta represents the value of beta
    % Recent revision by Liu Wei was on May 5th, 2020
    
    [visibleSize,n]=size(testdata);

    tic;
    W = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    %W1k = reshape(thetak(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    %W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
    b1 = theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);
    %b1k = thetak(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);
    b2 = theta(hiddenSize*visibleSize+hiddenSize+1:end);

    %cost=cost+sum(sum((theta-thetak).^2))/2;
    X=testdata;


    WTVB = W'*V+repmat(b2,1,n); 
    mWTVB  = max(WTVB,0);
    WXB = W*X + repmat(b1,1,n);
    sWTVB = smoothRELU(WTVB,mu);
    XsWTVB = X.* sWTVB;
    %sWXB = smoothRELU(WXB,mu);
    gWTVB = gradsmoothRELU(WTVB,mu);
    XgWTVB = X.* gWTVB;
    gWXB = gradsmoothRELU(WXB,mu);
    mXWTVB=mWTVB-XgWTVB;

    gradW = 2/n * V * mXWTVB'-beta*gWXB*X';
    gradV = beta + 2/n*W*mWTVB-2/n*W*XgWTVB;
    gradb1= -beta*sum(gWXB,2);
    gradb2= 2/n*sum(mXWTVB,2);
    tt=toc;

    cost = 1/n*sum(sum(mWTVB.^2))-2/n*sum(sum(XsWTVB));%+beta*sum(sum(V-sWXB));


    %grad=[gradW(:); gradb1(:); gradb2(:)];
    %gradb=[gradb1(:); gradb2(:)];
    %grad=grad+sum(theta-thetak);
end

function f = gradsmoothRELU(z,mu)
    %gradient of the smoothing function for RELU
    f=min(max(z/mu,0),1);
end

function f = smoothRELU(z,mu)
    %smoothing function for RELU
    f=zeros(size(z));
    f(z>=mu)=z(z>=mu)-mu/2;
    f(z<=0)=-1;
    index=(f==0);
    f(index)=z(index).*z(index)/2/mu;
    f(f<0)=0;
end

