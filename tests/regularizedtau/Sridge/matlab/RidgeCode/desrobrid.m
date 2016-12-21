 function [beta res sig edf w crit]=desrobrid(betin,lam,x,y,delsca,niter,tolcrit,tolres)
%DESROBRID [beta res sig edf w crit] =desrobrid(betin,lam,x,y,delsca,niter,tolcrit,tolres)
%Descent for RR-SE starting from initial estimate betin
%x,y= data,  lam= penalty
%Intercept always at the end
%delsca= "delta" of M-scale
%niter: max. iterations, default=100
%tolcrit,tolres: tolerances for criterion and residuals,
%         defaults 1.e-3 and1.e-4
%w=weights; lala=lambda modified by w
%crit=criterion

if nargin<6, niter=100; end
if nargin<7, tolcrit=1.e-3; end
if nargin<8, tolres=1.e-4; end
[n p]=size(x);
betinslo=betin(1:p); betinte=betin(p+1);
res0=y-x*betinslo-betinte; 
sig0=mscale(res0,0,delsca);         
%Iterations
iter=0; delta=inf; conve=inf;
crit0=n*sig0^2+lam*betinslo'*betinslo;  
binter=betinte;
while (iter<niter & (delta>tolcrit | conve>tolres))
    iter=iter+1;       
    tt=res0/sig0;
    w=weights(tt); rw=sqrt(w);
    ycen=y-binter;
    xw=x.*repmat(rw,1,p); yw=ycen.*rw;
    lala=mean(w.*tt.^2)*lam;
    xau=[xw; sqrt(lala)*eye(p)]; %augment x
    yau=[yw; zeros(p,1)];
    beta =xau\yau;  resin=y-x*beta;  
    binter=sum(resin .*w)/sum(w); 
    res=resin-binter; 
    sig=mscale(res,0,delsca);
    crit=n*sig^2+lam*beta'*beta;     deltold=delta;
    delta=1-crit/crit0; 
    conve=max(abs(res-res0))/sig; %measures convergence of residuals
    res0=res; sig0=sig; crit0=crit; 
end    
beta=[beta; binter];
hmat=xau*((xau'*xau)\xau'); h=diag(hmat); edf=sum(h(1:n));

function w=weights(r)   %Bisquare weights
w=(abs(r)<=1).*(1-r.^2).^2;

