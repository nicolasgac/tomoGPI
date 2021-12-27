function[J,J_MC,J_reg,dJ,dJ_MC,dJ_reg]=crit_TOMO8(f,g,iter)
%function [dJ,dJ_MC,dJ_reg_1,dJ_reg_2]=crit(f,g,nx,nz,nphi,q,qc1,qc2,qc3,lambda,z,mk,vk,nk);
% [dJ,dJ_MC,dJ_reg_1,dJ_reg_2]=dcrit1(f,g,nx,nz,nphi,q1,q2,lambda,z,mk,vk,nk);
% dJ=dJ_MC+lambda(1)*dJ_reg_1+lambda(2)*dJ_reg_2
% dJ_MC=-2*tranp(g-direct(f,nx,nz,nphi))
% dJ_reg_1=2*sum_j (1-q_j) [D^tD f]_j
% dJ_reg_2=sum_k sum_jk  2*(1/v_k^2)*(f_jk-m_k)
%
% Author: Nicolas Gac
% Date: 24/09/2016

% disp('');
% disp('Calcul de J_MC et DJ_MC...');


g_estimated=doProjection(iter,f);
dg=g-g_estimated;

[eam_g,eam_relatif_g,eqm_g,eqm_relatif_g]=sauvegarde_projection_TOMO8(dg,g,iter);

J_MC=sum(dg(:).^2);
df=doBackprojection(iter,dg);



dJ_MC=-2*df;
dJ=dJ_MC;
J_reg=0;
dJ_reg=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCUL DJ_reg
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ((getLambda(iter) ~= 0))
    
    switch(iter.methode)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Simple Gaussian  and  Generalized Gaussian
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      case 0:1
        ApplyLaplacianRegularization_to_dJ(iter,f,dJ,getLambda(iter),J_reg,normdJProjReg,getGradientIterationNb(iter),getOptimalStepIterationN(iter));
            dJ_reg=0;%RAJOUTER LA VRAIE VALEUR DE DJ_REG NICOLAS GAC 29/09/2016
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Cauchy
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 2
            %calcul de J_reg
            doLaplacian(iter,f,df);
            x=log(1+df.^2);
            J_reg=sum(x(:));
            J_reg=J_reg/length(f(:));
            
            %calcul de dJ_reg
            dJ_reg=zeros(size(f));
            dJ_reg=2*(df./(1+df.^2));
            dJ=dJ+getLambda(iter)*dJ_reg;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Huber
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 3
            %seuil=1
            %calcul de J_reg
            doLaplacien(iter,f,df);
            J_reg=sum(huber(df(:),iter.seuil_huber))/length(df(:));
            
            %size(Df(find(abs(Df)<iter.seuil_huber)))
            
            %calcul de dJ_reg
            x=DerivHuber(df(:),iter.seuil_huber);
            x=reshape(x,sampling.N_xn_FOV,sampling.N_yn_FOV,sampling.N_zn_FOV);
            doLaplacien(iter,x,dJ_reg);
            dJ=dJ+getLambda(iter)*dJ_reg;
        case 4
            %if (lambda ~= 0)
            %     disp('')
            %     disp('Calcul de J_reg et DJ_REG...');
            doLaplacien(iter,f,Df);
            %calcul de J_reg
            f_z=z.*(Df.^2);
            J_reg=sum(f_z(:));
            %J_reg=J_reg/length(f(:));
            
            %calcul de dJ_reg
            dJ_reg=single(zeros(size(f)));
            dJ_reg=2*z.*Df;
            
            
            %     disp('Calcul de J_reg et DJ_REG OK !!!');
            dJ=dJ+getLambda(iter)*dJ_reg;
        otherwise
            disp('Error(In crit_GCH_TOMO7)!!!!!!!!!!!!')
    end
    
    %     disp('Calcul de J_reg et DJ_REG OK !!!');
    %
    
else
    dJ_reg=0;
    J_reg=0;
end

J=J_MC+getLambda(iter)*J_reg;
clear g_estimated dg df;
