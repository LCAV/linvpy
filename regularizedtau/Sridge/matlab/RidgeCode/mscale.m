 function sig=mscale(x,normz,delta,tole)
%MSCALE sig=mscale(x,normz,delta,tole)
%sig=M-scale of x  
% sigma= solution of ave{rho(x_i/sigma)}=delta, where rho=bisquare
%delta: optional, default=0.5 
%tole optional, error tolerance,  default=1.e-5
%normz: optional; if >0 (default), normalize sig for consistency at the normal

if nargin<2, normz=1; end
if nargin<3, delta=0.5; end
if nargin<4; tole=1.e-5; end;

n=length(x); y=sort(abs(x));
n1=floor(n*(1-delta)); n2=ceil(n*(1-delta)/(1-delta/2));
qq=[y(n1) y(n2)];
u=rhoinv(delta/2);
sigin=[qq(1) qq(2)/u];  %initial interval
if qq(1)>=1, tolera=tole;   %relative or absolute tolerance, for sigma> or < 1
else, tolera=tole*qq(1);
end
if mean(x==0)>1-delta; sig=0;
else
   sig=fzero(@(sigma) averho(x,sigma,delta), sigin, optimset('TolX',tolera, 'Display','off'));
end
if normz>0, sig=sig/1.56;  %normalize
end

function r=rhobisq(x)  %Bisquare
r= 1-(1-x.^2).^3 .*(abs(x)<=1);

function aa=averho(x,sig,delta)
aa=mean(rhobisq(x/sig))-delta;

function x=rhoinv(u) %inverse function of rho
x=sqrt(1-(1-u)^(1/3));