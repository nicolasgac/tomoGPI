function [f_estimated_i_plus_1,iter_out]=gradient_GCH_TOMO8(f_estimated_i,f_real,g_real,iter)

% disp('****************************')
% disp('Descente de gradient... ')
% disp('****************************')

f_estimated_i_n=f_estimated_i;
for num_iter_gradient_n=1:1:getGradientIterationNb(iter)
    num_iter_gradient_n
    iter.num_iter=num_iter_gradient_n;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CALCUL DES J,J_reg et dJ, DJ_reg ....
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    
    [iter.J(iter.num_iter),iter.J_MC(iter.num_iter),iter.J_reg(iter.num_iter),dJ,dJ_MC,dJ_reg]=crit_TOMO8(f_estimated_i_n,g_real,iter);
    
  
    
    if (iter.save_file ~= 0)
        
        file_name=sprintf('%s/crit.dat',getOutputDirectory(iter));
        fid = fopen(file_name, 'a+b');
        
        fprintf(fid,['%d %f %f %f %f %f \n'], iter.num_iter-1,iter.J( iter.num_iter), ...
            iter.J_MC( iter.num_iter),iter.J_reg( iter.num_iter), ...
            iter.J_MC( iter.num_iter)/iter.J( iter.num_iter), ...
            iter.J_reg( iter.num_iter)*getLambda(iter)/ ...
            iter.J( iter.num_iter));
        
        fclose(fid);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % AFFICHAGE DES J,J_reg....
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %niter_done=size(iter.J,2);
    %if (niter_done > 1)
        %         if (iter.no_display_fig == 0)
        %             figure(4);
        %             plot(iter.J_MC(2:niter_done));
        %             title('J MC (20dB)');
        %             xlabel('iter');ylabel('J MC');drawnow;
        %             if (iter.lambda ~= 0)
        %                 figure(5);plot(iter.lambda*iter.J_reg(2:niter_done));title('lambda*J reg (20dB)');xlabel('iter');ylabel('lambda*J reg');drawnow;
        %             end
        %
        %
        %
        %             if (iter.lambda ~= 0)
        %                 figure(7); niter=[2:1:niter_done];
        %                 plot(niter,iter.J_MC(2:niter_done)./iter.J(2:niter_done),'-',niter,iter.lambda*iter.J_reg(2:niter_done)./iter.J(2:niter_done),'-.');xlabel('iter');legend('J MC/J','J reg/J');drawnow;
        %             end
        %
        %             figure(6);plot(iter.J(2:niter_done));title('J (20dB)');xlabel('iter');ylabel('J');drawnow;
        %
        %
        %
        %         end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CALCUL DU PAS
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %TODO tester plusieurs algo de descente de gradient : dicho,; gradient conjougué
        
        disp('')
        disp('Calcul de alpha...')
        
        %num_alpha=norm(dJ(:))^2;
        num_alpha=sum(dJ(:).^2)
        
          
    
        proj_dJ=doProjection(iter,dJ);
       
       
    
        %denum_alpha=norm(proj_dJ(:))^2;
        denum_alpha=2*sum(proj_dJ(:).^2)
        clear proj_dJ;
        
        %SI REGULARISATION
        if (getLambda(iter) ~= 0)
            s=zeros(size(dJ));
             doLaplacian(iter,dJ,s);
            s=sum(s(:).^2);
            denum_alpha=denum_alpha+getLambda(iter)*s;
        end
        
        
        
        alpha=num_alpha/denum_alpha;
        
        iter.alpha(iter.num_iter)=alpha;
        %     figure(3);plot(iter.alpha);title('pas');xlabel('iter');ylabel('alpha');drawnow;
        
        disp('Calcul de alpha OK !!!!')
        display(alpha);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MISE A JOUR DE f
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %alpha=0.03;
        %direction=-dJ;
        %clear dJ;
        %         size(f_estimated_i_n)
        %         size(alpha)
        %         size(direction)
        f_estimated_i_n=f_estimated_i_n-alpha.*dJ;
        clear dJ;
        %         g_max=max(g_real(:))
        %         g_min=min(g_real(:))
        %         f_max=max(f_estimated_i_n(:))
        %         f_min=min(f_estimated_i_n(:))
        
        % SAUVEGARDE DU VOLUME RECONSTRUIT TOUS LES iter.save_file
        iter=sauvegarde_volume_TOMO8(f_estimated_i_n,f_real,iter);
    
    end
    f_estimated_i_plus_1=f_estimated_i_n;
    iter_out=iter;
    
    %     disp('****************************')
    %     disp('Descente de gradient OK !!!!')
    %     disp('****************************')
end
