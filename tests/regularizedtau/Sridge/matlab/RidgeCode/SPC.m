function [lamda b mu scores]=SPC(x,cent); 
%SPC Spherical Principal Components (Locantore et al., 1999) [lamda b mu scores]=SPC(x,cent)
%lamda= Robust "eigenvalues" (increasing); b=Maatrix of eigenvectors
%mu=spatial mediana; scores=projection of x (centered) on eigenvectors
%If cent>0 (default),  x is centered
%cent=0 (no centgering) is used for "RobRidge"
if nargin<2, cent=1; end
[n p]=size(x);
[mu,w]=spamed(x,cent); xcen=centrar(x,mu');
y=xcen.*(w*ones(1,p)); 
[a s b]=svdecon(y); %uses "economic" SVD 
scores=xcen*b;  
%lamda=squared robust scales along prinipal directions
lamda=robsq(scores); 
[lamda ind]=sortrows(lamda); %sort lamdas increasing
b=b(:,ind); scores=scores(:,ind);

function [mu,w]= spamed(x,cent);     
%Spatial M-median , w=weights 
%If cent>0: mu=spatial median; else, mu=0
% w=1/||x-mu|| normalized
[n p]=size(x); del0=1.e-5;
if cent>0
     niter=20; tol=1.e-5; mu0=median(x)'; 
    dife=inf; ite=0;
    while ite<niter & dife>tol*p
        ite=ite+1;
        xc=centrar(x,mu0');
        w= sqrt(sum(xc'.^2))'; 
        deldis=del0*median(w);
        w=w.*(w>=deldis)+deldis*(w<deldis); %truncate just in case
        w=1./w;
        mu=sum((w*ones(1,p)).*x)'/sum(w); 
        dife=sqrt((mu0-mu)'*(mu0-mu)); 
        mu0=mu;
    end;
else,   mu=zeros(p,1);
        w= sqrt(sum(x'.^2))'; deldis=del0*median(w);
        w=w.*(w>=deldis)+deldis*(w<deldis); w=1./w;
end
w=w/sum(w);

function v=robsq(z)
%Squared ispersion
[n p]=size(z); mu=median(z);
zcen=centrar(z,mu);
v=zeros(p,1);
for k=1:p
    v(k)=mscale(zcen(:,k))^2;
end

function v=trimsq(y,alfa)
% alfa-trimmed squares v=trimsq(y,alfa)
%Not used
[n p]=size(y); m=floor(alfa*(n+1));
mu=median(y); z=centrar(y,mu).^2;
v=zeros(p,1);
for k=1:p
    u=sort(z(:,k)); v(k)=mean(u(1:m))/0.14;
end
