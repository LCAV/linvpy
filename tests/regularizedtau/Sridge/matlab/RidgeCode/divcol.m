function Y=divcol(X,sig)
%DIVCOL Divides columns of X by their std (default) o by row vector sig
[n p]=size(X); 
if nargin<2; divi=std(X);
else divi=sig;
end
Y=X./repmat(divi,n,1);
