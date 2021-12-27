function res = huber(dx,seuil)
% HUBER - Fonction L1L2 de pénalisation convexe 
%   
%   inputs : dx     vecteur des différences 
%            seuil  seuil entre le comportement quadratique et 
%                   comportement linéaire.
%
%   output : res    vecteur resultat de la fonction 


res = dx;

res(find(abs(dx)<seuil))=(res(find(abs(dx)<seuil))).^2;
res(find(abs(dx)>=seuil))=2*seuil*abs(res(find(abs(dx)>=seuil)))-seuil^2;
