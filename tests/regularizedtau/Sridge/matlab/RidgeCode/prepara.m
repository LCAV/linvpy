function [Xnor ycen mux sigx muy sigy]=prepara(X,y,robu)
% [Xnor ycen mux sigx muy sigy]=prepara(X,y, robu) 
%Centers y and the columns of X to zero location, and normalizes X to unit scale.
%If robu>0 (default): with robust M-estimates; else, means and SD
%Xnor=centered normalized X; ycen=centered y; mux=location vector of X;
%muy, sigy=location and scale of y
if nargin<3, robu=1; end
[n p]=size(X); 
if robu>0, mux=Mloca(X); Xnor=centrar(X,mux');
        sigx=zeros(1,p); muy=Mloca(y); ycen=y-muy;
        sigy=mscale(ycen);
        for j=1:p, sigx(j)=mscale(Xnor(:,j)); end
else, mux=mean(X)'; Xnor=centrar(X,mux');
        sigx=std(X); muy=mean(y); sigy=std(y); ycen=y-muy;
end
Xnor=divcol(Xnor,sigx);  
