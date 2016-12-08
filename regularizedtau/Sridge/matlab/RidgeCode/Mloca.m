function mu=Mloca(y,sig,efi)
%mu=Mloca(y,sig,efi)
%mu (column vector) = columnwise bisquare location M-estimator of matrix y. 
%sig (optional): scale, either a single number or a column vector.
%       Default=Normalized columnwise MAD
%efi (optional): desired efficiency, Choose 0.9 (default), 0.85 or 0.95
if nargin<2, sig=mad(y,1)'/0.675;
end
if nargin<3, efi=0.9; end
if efi==.85, kefi=3.44;
elseif efi==.9, kefi=3.88;
elseif efi==.95, kefi=4.68;
else, disp('wrong efi in Mloca')
end
[q n]=size(y);
niter=10;
nsig=size(sig,1);
if nsig==n, sigrep=repmat(sig',q,1);
else, sigrep=repmat(sig,q,n);    %f all sig's equal
end;
if q==1, mu=y;
else
    mume=median(y);    %inicial
    mu=mume;
    for j=1:niter
        z=(y-repmat(mu,q,1))./sigrep;
        w=bisq(z/kefi);
        sw=sum(w); nulos=(sw==0);  
        if sum(nulos)==0, mu=sum(y.*w)./sw; 
        else, nonul=~nulos;     %to avoid division by 0
            mu(nonul)=sum(y(:,nonul).*w(:,nonul))./sw(nonul); 
            mu(nulos)=mume(nulos);
        end
    end
end
mu=mu';

function w=bisq(z)  %bisquare weight function
t=z.^2; w=(t<=1).*(1-t).^2;