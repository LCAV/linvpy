function beta=desprepa(beta0, mux,sigx,muy)
%beta=desprepa(beta0, mux,sigx,muy)   inverse of "prepara" (returns to
%original scales)
%beta0 without intercept, beta has intercept at the end
beta=beta0./sigx'; betint=muy-mux'*beta; beta=[beta; betint]; 
