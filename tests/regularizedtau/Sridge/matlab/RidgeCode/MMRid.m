 function [beta res edf w mse]=MMRid(X,y,lam, betin,sigma,kefi,niter,tol)
%MMRID   [beta res edf w mse]=MMRid(X,y,lam, betin,sigma,kefi,niter,tol)
%RR-MM descent starting from initial estimate "betin", with given scale
%     "sigma" and penalty "lam"
%Minimizes criterion (k*sigma)^2*sum{rho(res/k*sigma)}+lamda*||beta1||^2
%Here rho''(0)=2
%niter=#(max. iterations),  default=50
%tol: tolerance for relative change of criterion and residuals, default=1.e-3;
%edf= equiv. deg fr.
%kefi= constant for efficiency, default =3.88 
%For the following efficiencies: %0.8   0.85   0.9   0.95,
% use kefi= 3.14   3.44   3.58   4.68
%res= residuals, w =weights.
%mse=estimated pred. error
%mse(j) for j=1:3 are based on: FPE, CV(n) and GCV(n) 
if nargin<6, kefi=3.88; end %eff.=90% 
if nargin<7, niter=50; end
if nargin<8, tol=1.e-3; end

[n p]=size(X);
kasig=kefi*sigma;
 betinte=betin(p+1); betinslo=betin(1:p);
res0=y-X*betinslo-betinte;
crit0=kasig^2*sum(rho(res0/kasig))+lam*norm(betinslo)^2;
%Iterations
iter=0; delta=inf; conve=inf;
binter=betinte;                    
while (iter<niter & (delta>tol | conve>tol))
    iter=iter+1;       
    tt=res0/kasig;
    w=weights(tt); rw=sqrt(w);
    ycen=y-binter;
    Xw=X.*repmat(rw,1,p); yw=ycen.*rw;
    Xau=[Xw; sqrt(lam)*eye(p)]; %augment X
    yau=[yw; zeros(p,1)];
    beta =Xau\yau;  resin=y-X*beta;  %here beta=slopes
    if sum(w)>0, binter=sum(resin .*w)/sum(w); 
    else, binter=median(resin);  
    end
    res=resin-binter;   % centered residuals
    crit=kasig^2*sum(rho(res/kasig))+lam*norm(beta)^2;
    deltold=delta;   delta=1-crit/crit0; 
    conve=max(abs(res-res0))/sigma; %measures convergence of residuals
    res0=res; crit0=crit; 
end    
beta=[beta; binter];
hmat=Xau*((Xau'*Xau)\Xau'); h=diag(hmat); edf=sum(h(1:n));
%Three versions of MSE:
%1: FPE
aa=mean(psibis(res/kasig).^2); bb=mean(psipri(res/kasig));
    if(bb<0.001), disp('MMrid'), disp([aa bb]), end
mse1=sigma^2*(mean(rho(res/kasig))+ edf*aa/(n*bb) );
%2:  approximate leave-one-out CV
D=diag(psipri(res/kasig)); H=(X'*D)*X; U=H+2*lam*eye(p);
h=diag(X*(U\X')); hpri=psipri(res/kasig).*h;
kapsi=kasig*psibis(res/kasig);
resin=res+h.*kapsi./(1-hpri); %resid CV
ktau=5;  %coonstant for tau-scale
mse2=tauscale(resin,ktau).^2;
%3: the same, with GCV
hpri=mean(hpri);
resin=res+h.*(kasig*psibis(res/kasig))/(1-hpri); %resid CV
mse3=tauscale(resin,ktau).^2;
       
mse=[mse1 mse2 mse3];
  

function r=rho(x)  %Bisquare
r= (1-(1-x.^2).^3 .*(abs(x)<=1))/3;  %to make rho''(0)=2
function z=psibis(r)   %psibis=rho'
    z=2*(abs(r)<=1).*r.*(1-r.^2).^2; %psi bisquare
function z=psipri(r)     %psipri=psibis'
    z=2*(abs(r)<=1).*(1-r.^2).*(1-5*r.^2);
function w=weights(r)   %w=(psibis(r)/r)/2 to eliminate the "2" from 2*lam
w=(abs(r)<=1).*(1-r.^2).^2;
