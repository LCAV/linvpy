function [y ind]=unitol(x,eps)
%y=x(ind)
%Same as function "unique" but with tolerance eps
if nargin<2, eps=1.e-10; end
n=length(x); 
[y ind]=sort(x); ydif=diff(y);  
z=y(2:n); ii=ind(2:n); 
y=[y(1);z(ydif>eps)]; ind=[ind(1); ii(ydif>eps)];