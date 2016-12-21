The main procedure is RobRidge

%[beta resid edf lamin]= RobRidge(X,y, numlam,cualcv, showhist,nkeep)
%Solves n*sig^2 *sum{rho(resid/sig)+lam*||beta1||^2} = min
% Required input: X, y data set
%
% Optional input:
% numlam: number of lambda values, default =min(n,p,20)
% cualcv: method for estimating prediction error. If cualcv=0 (default): approximate leave-one-out CV; 
%     if >0: actual cualcv--fold CV ("N_{lambda})";
%     if 0<cualcv<1: random test sample of size n*cualcv
% showhist: if >0, print edf and mse for eadh lambda (default=0)
% nkeep= number of candidates to be kept for full iteration in the Peña-Yohai procedure (default=5)
%
% Output
% beta= (p+1)-vector of regression parameters, with beta(p+1)=intercept
% resid= residual vector
% edf= final equivalent degrees of freedom ("p hat")
%lamin= optimal lambda
