function [betaSE betaMM residSE residMM edfSE edfMM mseMM sigma]=RidSEMM(X,y,lam,deltaesc,nkeep,keff)
%[betaSE betaMM residSE residMM edfSE edfMM mseMM]=RidSEMM(X,y,lam,deltaesc,nkeep,keff)
%Computes RR-SE and RR-MM for a given lambda=lam
%X,Y= data, X is assumed normalized
%deltaesc (optional)= "delta" for scale M-estimate (default: =0.5)
%nkep (optional)= # candidates for full iteration in Peña-YOhai 
%       (default=5)
% keff (optional): constant "c" for efficiency from lilst below:
% efipos=[0.8 0.85 0.9 0.95]; %possible efficiencies
% keff= [3.14 3.44 3.88 4.68]; %sus constantes
% default: efficiemcy = nominal 0.85 adjusted by edf
%
% betaSE, betaMM=estimates, with intercept at the end
% residSE, residMM= residuals; edfSE, edfMM= equivalent d.f.
% mseMM= estimated robust MSE
if nargin<4, deltaesc=0.5; end
if nargin<5, nkeep=5; end
%Compute Peña-Yohai; sigma is already corrected for dimension
[betaSE residSE sigma edfSE]=PeYoRid(X,y,lam,deltaesc,nkeep);
if nargin<6     %correccion
    psn=edfSE/length(y);
    if psn<0.1, keff=3.5;
    elseif psn<0.2, keff=3.7;
    elseif psn<0.33, keff=4;
    else, keff=4.2;
    end
end
[betaMM residMM edfMM w mseMM]=MMRid(X,y,lam, betaSE,sigma,keff);
