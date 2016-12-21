function [beta,resid,sigma,edf,lamin]= sridge(X,y,numlam, cualcv,showhist,nkeep,niter)                 
%sridge  Calculates the s-ridge regression estimate 
%        adapted from Ricardo A. Maronnas matlab code from "Robust ridge regression for high dimensional
%        data" (2012)
%
%   Inputs: 
%               X - Regression Matrix [N x P]
%               y - respones [N x 1]
%               numlam - number of lambda values, default =min(n,p,20)
%               cualcv - method for estimating prediction error. 
%                        cualcv--fold CV ("N_{lambda}";
%               showhist - if >0, print edf and mse for eadh lambda (default=0)
%               nkeep - number of candidates to be kept for full iteration in the Pe?a-Yohai procedure (default=5)
%               niter - maximum number of iteration steps for the
%                          s-ridge calculation [1 x 1]
%
%   Outputs:
%               beta - (p+1)-vector of regression parameters, %beta(1)=intercept
%               resid - residual vector
%               edf - final equivalent degrees of freedom ("p hat")
%               lamin - optimal lambda

disp(X)
disp(y)
n=size(X,1); 
if nargin<3, numlam=min([20 n size(X,2)]); end
if nargin<4, cualcv=1; end
if nargin<5, showhist=0; end
if nargin<6, nkeep=5; end

% %Normalize and center X and y
% [Xnor,ynor,mux,sigx,muy]=prepara(X,y);
Xnor=X;
ynor=y;
%Spherical Principal COmponents (no centering)
%privar, Beig= vector of robust "eigenvalues" and matrix of eigenvectors
%Xnor is now =PCA scores= "orthonormalized Xnor "
[privar,Beig, ~, Xnor]=SPC(Xnor,0); 
[n,p]=size(Xnor);  %p is now the "actual" dimension
privar=privar*n; %Makes the robust eigenvalues of the same order as those of classical PCA used for LS
 nlam=min([p numlam]);  
pmax=min([p n/2]);   %edf<=n/2 to keep BDP >=0.25
pp=linspace(1,pmax,nlam);  %"candidate edf's"
lamdas=findlam(privar,pp); %find lambdas corresponding to the edf's
deltas=0.5*(1-pp/n);  %for the M-escale used with Pe?a-Yohai
msemin=inf; historia=[];

 %Actual CV, or test sample
    [~,orden]=sort(randn(n,1));  %Random permutation
  for klam=1:nlam
      [klam;nlam]
        lam=lamdas(klam); deltaesc=deltas(klam);
        mse=CVRidRob(Xnor,ynor,cualcv,orden,lam,pp(klam));
        if mse<msemin,  msemin=mse; lamin=lam; delmin=deltaesc; end
        historia=[historia; [lam mse]];
  end
    [beta,resid,sigma,edf]=PeYoRid(Xnor,ynor,lamin,delmin,nkeep,niter);

if showhist>0,  disp(historia), end
%Denormalize beta
betaslo=Beig*beta(1:p); bint=beta(p+1); 
% beta=desprepa(betaslo, mux,sigx,muy+bint); 
beta=[bint;betaslo];

%put intercept to the beging of the vector
% beta=[beta(end);beta(1:end-1)];


