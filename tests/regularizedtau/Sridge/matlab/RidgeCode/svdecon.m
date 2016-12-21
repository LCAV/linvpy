function [a s b]=svdecon(x) 
%[a s b]=svdecon(x), x=a*diag(s)*b' 
%Economic SVD economica: a=nxq and b=pxq with q=rank(x); 
%s=q-vector, decreasing
[n p]=size(x); q=max(n,p);
[a s b]=svd(x,0); 
if p>n, s=s(:,1:n); %now s and b are nxn
    b=b(:,1:n);
end
epsil=1.e-8; s=diag(s); 
sm=q*s(1)*epsil; %omit null singular values
a=a(:,s>sm); b=b(:,s>sm); s=s(s>sm);