function [f_estimated_i_plus_1,iter_out,lambda_calcul]=gradient_JMAP_TOMO8(f_estimated_i,f_real,g_real,iter,z,lambda_n,lambda_calcul,num_iter_global_I)

% disp('****************************')
% disp('Descente de gradient... ')
% disp('****************************')

f_estimated_i_n=f_estimated_i;

%% 2eme boucle
for num_iter_gradient_n=1:1:getGradientIterationNb(iter)
    
    iter.num_iter=(num_iter_global_I-1)*getGradientIterationNb(iter)+num_iter_gradient_n;
    num_iter=iter.num_iter
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CALCUL DES J,J_reg et dJ, DJ_reg ....
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [iter.J(iter.num_iter),iter.J_MC(iter.num_iter),iter.J_reg(iter.num_iter),dJ,dJ_MC,dJ_reg]=crit_TOMO7(f_estimated_i_n,g_real,iter,z,lambda_n);
    %         iter.J(iter.num_iter)
    %         iter.J_MC(iter.num_iter)
    %         lambda*iter.J_reg(iter.num_iter)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % AFFICHAGE DES J,J_reg....
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %     iter.J
    %     iter.J_MC
    %     iter.J_reg
    %     lambda()*iter.J_reg
    
    if (iter.save_file ~= 0)
        
        file_name=sprintf('%s/crit.dat',iter.repertoire_test);
        fid = fopen(file_name, 'a+b');
        
        fprintf(fid,['%d %f %f %f %f %f \n'],num_iter-1,iter.J(num_iter), ...
            iter.J_MC(num_iter),iter.J_reg(num_iter), ...
            iter.J_MC(num_iter)/iter.J(num_iter), ...
            iter.J_reg(num_iter)*iter.lambda/ ...
            iter.J(num_iter));
        
        fclose(fid);
    end
    
    niter_done=size(iter.J,2);
    if (niter_done > 1)
        %         if iter.no_display_fig==0
        %             figure(4);plot(iter.J_MC(2:niter_done));title('J MC');xlabel('iter');ylabel('J MC');drawnow;
        %         end
        %if (lambda_n ~= 0)
        lambda_calcul(num_iter_gradient_n,num_iter_global_I)=lambda_n;
        lambda_Jreg=zeros(1,niter_done-1);
        lambda_calcul_v=lambda_calcul(:);
        lambda_calcul_v=lambda_calcul_v';
        lambda_Jreg=(lambda_calcul_v(2:niter_done)).*iter.J_reg(2:niter_done);
        
        %         if iter.no_display_fig==0
        %             figure(5);plot(lambda_Jreg);title('lambda*J reg');xlabel('iter');ylabel('lambda*J reg');drawnow;
        %         end
        %end
        
        %if (lambda_n ~= 0)
        %         if iter.no_display_fig==0
        %             figure(7);
        %             niter=2:1:niter_done;
        %             plot(niter,iter.J_MC(2:niter_done)./iter.J(2:niter_done),'-',niter,lambda_Jreg./iter.J(2:niter_done),'-.');xlabel('iter');legend('J MC/J','J reg/J');drawnow;
        %         end
        %end
        
        %         if iter.no_display_fig==0
        %             figure(6);plot(iter.J(2:niter_done));title('J');xlabel('iter');ylabel('J');drawnow;
        %         end
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CALCUL DU PAS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %TODO tester plusieurs algo de descente de gradient : dicho,; gradient conjougué
    
    %     disp('')
    %     disp('Calcul de alpha...')
    
    %num_alpha=norm(dJ(:))^2;
    num_alpha=sum(dJ(:).^2);
    
    proj_dJ=doProjection(a,dJ); % Train the class - dummy method   
 
    
    %denum_alpha=norm(proj_dJ(:))^2;
    denum_alpha=sum(proj_dJ(:).^2);
    
    clear proj_dJ;
    %SI REGULARISATION
    %     if (lambda_n ~= 0)
    %         s=convn(dJ,iter.D,'same');
    %         s=sum(s(:).^2);
    %         denum_alpha=denum_alpha+lambda_n*s;
    %     end
    
    if (lambda_n ~= 0)
        %iter.lambda
        Y=ones(sampling.N_xn_FOV,sampling.N_yn_FOV,sampling.N_zn_FOV);
        Y=z.^0.5;
        Y_dJ=Y.*dJ;
        s=sum(Y_dJ(:).^2);
        denum_alpha=denum_alpha+lambda_n*s;
    end
    
    alpha=0.5*num_alpha/denum_alpha;
    
    iter.alpha(iter.num_iter)=alpha;
    
    %     if iter.no_display_fig==0
    %         figure(3);plot(iter.alpha);title('pas');xlabel('iter');ylabel('alpha');drawnow;
    %     end
    
    %     disp('Calcul de alpha OK !!!!')
    alpha
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MISE A JOUR DE f
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %alpha=0.004;
    
    %direction=-dJ;
    %clear dJ;
    %     size(f_estimated_i_n)
    %     size(alpha)
    %     size(direction)
    f_estimated_i_n=f_estimated_i_n-alpha*dJ;
    %     if iter.no_display_fig==0
    %         figure(2);imagesc(f_estimated_i_n(:,:,sampling.N_zn_FOV/2));colormap(gray);title(['Image gradient (Iter grad=',num2str(iter.num_iter),'Iter=',')']);colorbar;drawnow;
    %     end
    % SAUVEGARDE DU VOLUME RECONSTRUIT TOUS LES iter.save_file
    iter=sauvegarde_volume_TOMO8(f_estimated_i_n,f_real,iter);
end

f_estimated_i_plus_1=f_estimated_i_n;
iter_out=iter;

% disp('****************************')
% disp('Descente de gradient OK !!!!')
% disp('****************************')
