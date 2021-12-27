function res = DerivHuber(dx,seuil)
% DERIVHUBER - Derive de la fonction de Huber 
%   
%   inputs : dx     vecteur des diff�rences ou image des differences 
%            seuil  seuil entre le comportement quadratique et 
%                   comportement lin�aire.
%
%   output : res    vecteur resultat de la fonction 


res = dx;
res(find(dx<=-seuil))=-2*seuil;
res(find(abs(dx)<seuil))=2*(res(find(abs(dx)<seuil)));
res(find(dx>=seuil))=2*seuil;



