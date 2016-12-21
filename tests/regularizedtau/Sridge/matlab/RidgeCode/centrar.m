function Y=centrar(X,mu)
%Y=centrar(X,mu): Centers matrix X by means (default) o by row vector mu
[n p]=size(X);
if nargin<2; centro=mean(X);
else centro=mu;
end
Y=X-repmat(centro,n,1);