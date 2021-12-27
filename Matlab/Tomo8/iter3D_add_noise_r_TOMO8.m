function [b,y,rsb_dB_out]=iter3D_add_noise_r_TOMO8(y,rsb_dB_in);
%add_noise_r ajoute un bruit blanc gaussian sur y
%            y en entree est le signal sans bruit
%            y en sortie est le signal avec bruit
%            rsb_dB_in en entree est le rapport s/n demande en dB
%            rsb_dB_out en sortie est le rapport s/n effectivement obtenu en dB 
%            b est la sequence du bruit 
% function [b,y,rsb_dB]=add_noise_r(y,rsb_dB);
%----------------------------------------------------------------------------%  Auteur         : Ali Djafari
%  Mise a jour le : 26-03-1991
%----------------------------------------------------------------------------     
 rsb_in=rsb_dB_in/10;
rsb_in=10^rsb_in;

  [ly,ky,zy]=size(y);
  s=y(:);
 
  vs=cov(s);
   %disp([' Variance du signal avant bruit = ',num2str(mean_s)]);
   
   %disp([' Mean du signal avant bruit = ',num2str(vs)]);
  a=sqrt(vs/rsb_in);

 b=randn(size(s));
 b=a*b;
vb=cov(b);

 rsb_dB_out=10*log10(vs/vb);
  
  disp([' Variance du signal = ',num2str(vs)]);
  disp([' Variance du bruit  = ',num2str(vb)]);
  disp([' Rapport S/N        = ',num2str(rsb_dB_out)]);

  b=reshape(b,ly,ky,zy);
  y=y+b;
return
