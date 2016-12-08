function mse =CVRidRob(XX,yy,nfold,orden,lam,gradlib)
%mse=CVRidRob(XX,yy,nfold,orden,lam,gradlib)
%XX,yy= data
%If nfold>1, performs nfold-CV 
%If 0<nfold<1: test sample of size nfold*n
%gradlib= degrees of freedom = edf
%orden= vector of random ordering
%mse= resulting robust MSE

%Reorder data
X=XX(orden,:); y=yy(orden);
%Segment data
[n p]=size(X); indin=1:n;  
if nfold>=2
    nestim=n*(1-1/nfold); %  #(Xesti)
                       lamcv=lam;
    deltaesc=0.5*(1-gradlib/nestim);
    inint=floor(linspace(0,n,nfold+1)); 
    resid=zeros(n,1);
    for kk=1:nfold
        testk=(inint(kk)+1):inint(kk+1);
        estik=setdiff(indin,testk);
        Xtest=X(testk,:); Xesti=X(estik,:);
        ytest=y(testk); yesti=y(estik);
        [betaSE,~,~,~]=PeYoRid(Xesti,yesti,lamcv,deltaesc);
        beta=betaSE(1:p); bint=betaSE(p+1); 
        fitk=Xtest*beta+bint; resid(testk)=ytest-fitk;
    end
else
    ntest=floor(n*nfold);   nestim=n-ntest;
    lamcv=lam;
    deltaesc=0.5*(1-gradlib/nestim);
    if ntest<5 | ntest>=n, disp('wrong nfold in CVRidRob'), end
    Xtest=X(1:ntest,:); ytest=y(1:ntest);
    Xesti=X(ntest+1:n,:); yesti=y(ntest+1:n);
    [betaSE,~,~,~]=PeYoRid(Xesti,yesti,lamcv,deltaesc);
    beta=betaSE(1:p); bint=betaSE(p+1); 
    fit=Xtest*beta+bint; resid=ytest-fit;
end
ktau=5;
mse=tauscale(resid,ktau).^2;

