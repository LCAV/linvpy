function lamr= findlam(vals,r)
%FINDLAM lamr= findlam(vals,r)   column vector
%Finds lamdas which yield edf=r
p=length(vals); nr=length(r);     
lamr=zeros(nr,1); 
lam1=0;  lam2=max(vals)*(p./r-1);   
for i=1:nr
    lam0=[lam1 lam2(i)+0.5];   %the value 0.5 is for lam=0
   lamr(i)=fzero(@(lam) sumrat(lam,vals,r(i)),lam0, optimset('Display','off'));
end
  
function susu=sumrat(lam,vals,r)
    susu=sum(vals./(vals+lam))-r;           
 