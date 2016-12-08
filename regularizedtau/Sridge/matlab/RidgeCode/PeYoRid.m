 function [betamin resid sigma edf pesos]=PeYoRid(X,y,lam,deltaesc,nkeep,niter,tolcrit,tolres)
%[betamin resid sigma edf pesos]=PeYoRid(X,y,lam,deltaesc,nkeep,niter,tolcrit,tolres)
%Peña-Yohai for RR
%X e y assumed normalized. 
%Intercept added at the end
%niter=# final iterations; 
% tolcrit, tolres=tolerances for relative change in criterion and residuals
%pseos= final weights
if nargin<5, nkeep=5; end
if nargin<6, niter=50; end
if nargin<7, tolcrit=1.e-3; end
if nargin<8, tolres=1.e-4; end
     
factr=facon(deltaesc);  %corrects lambda to match RR-LS
lamfac=factr*lam;
 [n p]=size(X); Xuno=[X ones(n,1)];
singu=0;  %If too much collinearity, choose singu=1 
[betals Xau yau]=regrid(X,y,lam); %RR-LS
reslau=yau-Xau*betals;
[A sval B]=svdecon(Xau); 
h=sum(A'.^2)'; w=(reslau./(1-h)).^2;
ab=A*B'; Q=ab'*diag(w)*ab;
[U d]=eig(Q);
Z=A*B'*U; Z=Z(1:n,:); %take out added "observations"
    
sigls=mscale(reslau(1:n),0,deltaesc);
critls=n*sigls^2+lamfac*norm(betals(1:p))^2;
critkeep=critls; betakeep=betals;

 %prosac= proportion of extreme residuals to be omitted
prosac=deltaesc;
m=round(prosac*n+1.e-6);
n1=n-m; cuales=0; 
lam1=lam*n1/n;   %recall that thera are only n1 "actual" obs. 
% sigmin=sigls; beta=zeros(p,1);
for j=1:size(Z,2)
    zj=Z(:,j); [zjord ii]=sort(zj);          
    Xj=X(ii,:); yord=y(ii);
    Xjj=Xj(1:n1,:); yj=yord(1:n1);  %higher
    betaj=regrid(Xjj,yj,lam1); resj=y-Xuno*betaj;
    sigj=mscale(resj,0,deltaesc);  critj=n*sigj^2+lamfac*norm(betaj(1:p))^2;   
    critkeep=[critkeep; critj];  betakeep=[betakeep betaj]; 
    Xjj=Xj(m+1:n,:); yj=yord(m+1:n);  %lower
    betaj =regrid(Xjj,yj,lam1); resj=y-Xuno*betaj;
    sigj=mscale(resj,0,deltaesc);  critj=n*sigj^2+lamfac*norm(betaj(1:p))^2;
    critkeep=[critkeep; critj];  betakeep=[betakeep betaj]; 
    [zjabord ii]=sort(abs(zj)); yord=y(ii);       
    Xj=X(ii,:); 
    Xjj=Xj(1:n1,:); yj=yord(1:n1);  %higher absl. vals.
    betaj=regrid(Xjj,yj,lam1); resj=y-Xuno*betaj;         
    sigj=mscale(resj,0,deltaesc);  critj=n*sigj^2+lamfac*norm(betaj(1:p))^2;  
    critkeep=[critkeep; critj];  betakeep=[betakeep betaj]; 
    nk=length(critkeep);
    if nk>nkeep         %omit unneeded candidates
        [critor ii]=unitol(critkeep); %omit repeated ones, and sort
        betakeep=betakeep(:,ii);
        nuni=min([nkeep length(ii)]);
        critkeep=critor(1:nuni); betakeep=betakeep(:,1:nuni);
    end
end
          
%Descent
critmin=inf; 
if nkeep>size(betakeep,2)
    nkeep=size(betakeep,2);
end
for k=1:nkeep
    betak=betakeep(:,k);          
    [betak resk sigmak edfk pesok critk]=desrobrid(betak,lamfac,X,y,deltaesc,niter,tolcrit,tolres);
     if critk<critmin, critmin=critk; sigma=sigmak; betamin=betak; resid=resk; 
         edf=edfk; pesos=pesok;
     end
end
         
deltult=0.5*(1-(edf+1)/n);  % "delta" for para MM
deltult=max([deltult 0.25]);
%c0= consant. for consistency
c0=7.8464-34.6565*deltult + 75.2573*deltult^2 -62.5880*deltult^3;
sigma=mscale(resid,0,deltult)/c0;    
q=(edf+1)/n; k1=1.29; k2= -6.02; corrige=1/(1-(k1+k2/n)*q);
sigma=sigma*corrige;  %bias corrección 

 function beta=regresing(X,y,singu)
%If X collinear, use singu>0
if nargin<3, singu=0; end
if singu>0, beta=pinv(X)*y;
else, beta=X\y;
end
        
function [beta Xau yau]=regrid(X,y,lam,delta)  %RR-LS
[n p]=size(X);
Xau=[[X ones(n,1)]; [sqrt(lam)*eye(p) zeros(p,1)]]; %aumentar
yau=[y; zeros(p,1)]; beta=Xau\yau; 



    function ff=facon(delta)
    %factor that corrects lambda to match RR-LS
    ff=23.9716 -73.4391*delta+ 64.9480*delta^2;