function sigmas=tauscale(x,ktau,delta)  
% tau scales (row vector) of x for several constants ktau (row)
%delta= "delta" for initial M-scale, default=0.5
if nargin<3, delta=0.5; end
sigmas=[]; s0=mscale(x,0,delta);
%constant for consistency of s0
c0=7.8464-34.6565*delta + 75.2573*delta^2 -62.5880*delta^3;
s0=s0/c0;
for k=ktau
    romed=mean(rho(x/(s0*k)));  sig=k*s0*sqrt(romed);
    sigmas=[sigmas sig];
end

 function r=rho(x)  %Bisquare
r= (1-(1-x.^2).^3 .*(abs(x)<=1))/3;  %para que rho''(0)=2