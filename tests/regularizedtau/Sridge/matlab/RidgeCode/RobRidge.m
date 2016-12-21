function [beta resid sigma edf lamin]= RobRidge(X,y,numlam, cualcv,showhist,nkeep)                 
%ROBRIDGE [beta resid edf lamin]= RobRidge(X,y, numlam,cualcv, showhist,nkeep)
%Solves n*sig^2 *sum{rho(resid/sig)+lam*||beta1||^2} = min
% Required input: X, y data set
% Optional input:
% numlam: number of lambda values, default =min(n,p,20)
% cualcv: method for estimating prediction error. If cualcv=0 (default): approximate leave-one-out CV; 
%     if >0: actual cualcv--fold CV ("N_{lambda}";
%     if 0<cualcv<1: random test sample of size n*cualcv
% showhist: if >0, print edf and mse for eadh lambda (default=0)
% nkeep= number of candidates to be kept for full iteration in the Peña-Yohai procedure (default=5)
% Output
% beta= (p+1)-vector of regression parameters, %beta(p+1)=intercept
% resid= residual vector
% edf= final equivalent degrees of freedom ("p hat")
%lamin= optimal lambda

[n p]=size(X); 
if nargin<3, numlam=min([20 n size(X,2)]); end
if nargin<4, cualcv=1; end
if nargin<5, showhist=0; end
if nargin<6, nkeep=5; end

%Normalize and center X and y
[Xnor ynor mux sigx muy]=prepara(X,y);
%Spherical Principal COmponents (no centering)
%privar, Beig= vector of robust "eigenvalues" and matrix of eigenvectors
%Xnor is now =PCA scores= "orthonormalized Xnor "
[privar Beig muspam Xnor]=SPC(Xnor,0); 
[n p]=size(Xnor);  %p is now the "actual" dimension
privar=privar*n; %Makes the robust eigenvalues of the same order as those of classical PCA used for LS
 nlam=min([p numlam]);  
pmax=min([p n/2]);   %edf<=n/2 to keep BDP >=0.25
pp=linspace(1,pmax,nlam);  %"candidate edf's"
lamdas=findlam(privar,pp); %find lambdas corresponding to the edf's
deltas=0.5*(1-pp/n);  %for the M-escale used with Peña-Yohai
msemin=inf; historia=[];

 %Actual CV, or test sample
    [tt orden]=sort(randn(n,1));  %Random permutation
  for klam=1:nlam
        lam=lamdas(klam); deltaesc=deltas(klam);
        mse=CVRidRob(Xnor,ynor,cualcv,orden,lam,pp(klam));    
        if mse<msemin,  msemin=mse; lamin=lam; delmin=deltaesc; end
        historia=[historia; [lam mse]];
  end
    [beta resid sigma edf]=PeYoRid(X,y,lam,deltaesc,nkeep);

if showhist>0,  disp(historia), end
%Denormalize beta
betaslo=Beig*beta(1:p); bint=beta(p+1); 
beta=desprepa(betaslo, mux,sigx,muy+bint); 


