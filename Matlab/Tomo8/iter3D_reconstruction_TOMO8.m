function [f_estimated]=iter3D_reconstruction_TOMO8(rep)


iter = Iter3D(rep); % Create a class instance



iter.methode=5;

iter.save_file=1;
iter.save_volume=2;
iter.fc=1.0;




f_estimated=CreateVolumeInit(iter);
f_real=CreateVolumeReal(iter);
%f_real=single(zeros(getXVolumePixelNb(iter),getXVolumePixelNb(iter),getZVolumePixelNb(iter)));
g_real=getSinoReal(iter);
%[g_real,rsb_in]=addNoise(iter,g_real);
% file_name=sprintf('Projections_%d/P_ER_GPU_NOISE_%2.1fdB.s',size(g_real,3),rsb_in);
%       fid = fopen(file_name, 'wb');
%       fwrite(fid,g_real ,'float');
%       fclose(fid);


        %g_filtred=doFilter_ramp(iter,g_real,iter.fc);
        %file_name=sprintf('Projections_%d/P_ER_GPU_FILTERED_f%1.2f_NOISE_%2.1fdB.s',size(g_real,3),iter.fc,rsb_in);
        %fid = fopen(file_name, 'wb');
        %fwrite(fid,g_filtred ,'float');
        %fclose(fid);

f_FDK=doFDK(iter,g_real,iter.fc);
    file_name=sprintf('%s/volume_FDK.v',getOutputDirectory(iter));
        fid = fopen(file_name, 'wb');
        fwrite(fid,f_FDK ,'float');
        fclose(fid);
        
       
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Fonction Matlab pour reconstruction itérative en tomo X utilisant les projecteurs et retroprojecteurs GPUs
%
% GPI
% NIcolas GAC
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
%%
%iter.methode=0;
switch(iter.methode)
    case 0
        disp('ITER_GAUSSIEN');
    case 1
        disp('ITER_GENERALIZED_GAUSSIEN');
    case 2
        disp('ITER_CAUCHY');
    case 3
        disp('ITER_HUBER');
    case 4
        disp('ITER_JMAP');
    otherwise
        disp('error(methode not exist)!!!!!!!!!!!!!!!!!!!!');
        
end


switch(iter.methode) 
    case {0,1,2,3} % Simple Gaussian Generalized Gaussian  Cauchy 
                   [f_estimated,iter]=gradient_GCH_TOMO8(f_estimated,f_real, g_real,iter);
                   %doGradient(iter,f_estimated,g_real);
    case 4 % JMAP
        niter_global_I=getGlobalIterationNb(iter);
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   INITIALISATION DE f,z,v_psi,v_f   %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha_psi=2.1;
        beta_psi=0.11;
        alpha_f=3;
        beta_f=2;
        
        alpha_z=1;
        beta_z=1;
        
        v_psi_ini=beta_psi/(alpha_psi-1);
        v_f_ini=beta_f/(alpha_f-1);
        
        %N=sampling.N_xn_FOV*sampling.N_yn_FOV*sampling.N_zn_FOV;
        M=getUSinogramPixelNb(iter)*getVSinogramPixelNb(iter)*getProjectionSinogramPixelNb(iter);
        
        z_ini=single(ones(getXVolumePixelNb(iter),getYVolumePixelNb(iter),getZVolumePixelNb(iter)));
        z=z_ini;
        
        v_psi=v_psi_ini;
        v_f=v_f_ini;
        
        lambda=single(zeros(iter.niter_global_I+1,1));
        lambda(1)=v_psi_ini/v_f_ini;
        lambda_n=lambda(1);
        
        lambda_calcul=single(zeros(iter.niter_gradient_N,iter.niter_global_I));
        
        %%
        
        for num_iter_global_I=1:1:niter_global_I
            num_iter_global_I
            [f_estimated,iter,lambda_calcul]=gradient_JMAP_TOMO8(f_estimated,f_real,g_real,iter,z,lambda_n,lambda_calcul,num_iter_global_I);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % MISE A JOUR DE v_psi
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            v_psi=(0.5*iter.J_MC(iter.num_iter)+beta_psi)/(alpha_psi+M/2+1)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % MISE A JOUR DE v_f
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %     f_z=z.*(f_estimated_i_n.^2);
            %     v_f=(beta_f+0.5*sum(f_z(:)))/(alpha_f+N/2+1)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % MISE A JOUR DE z
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %z_c=zeros(z_lign,1);
            %z_c_n=diag(z);
            z=(alpha_z-0.5)./(beta_z+((f_estimated.^2)/(2*v_f)));
            if (mod(iter.num_iter,200)==0)
                if (iter.save_file ~= 0)
                    file_name=sprintf('%s/z_%03d.v',getOutptDirectory(iter),iter.num_iter);
                    fid = fopen(file_name, 'wb');
                    fwrite(fid,z,'float');
                    fclose(fid);
                end
            end
            
            lambda_n=v_psi/v_f;
            lambda(num_iter_global_I+1)=lambda_n;
            
            if (iter.save_file ~= 0)
                
                file_name=sprintf('%s/hyper.dat',getOutptDirectory(iter));
                fid = fopen(file_name, 'a+b');
                
                fprintf(fid,'%d %f %f %f %f  \n',num_iter_global_I,v_psi,v_f,lambda_n,z(128,128,128));
                
                fclose(fid);
            end
        end
    otherwise
        disp('error(In iter3D_reconstruction_TOMO8)!!!!!!!!!!!!!!!!!!!!!!')
end

%%

%f_chapeau=f_estimated;
%g_chapeau=single(zeros(sampling.N_un,sampling.N_vn,sampling.N_phi));
%projection3D_TOMOX_mex(f_chapeau,g_chapeau,kernel_P,constante_GPU, sampling,iter.projecteur);

%delta_f=((sum((f_chapeau(:)-f_real(:)).^2)))/((sum((f_real(:)).^2)));
%delta_g=((sum((g_chapeau(:)-g_real(:)).^2)))/((sum((g_real(:)).^2)));
%iter.lambda
switch(iter.methode)
    case 0
        % $$$         if(iter.save_file ~= 0)
        % $$$              file_name=sprintf('%s/delta_gaussian.dat',iter.repertoire_test);
        % $$$              fid=fopen(file_name,'a+b');
        % $$$              fprintf(fid,'%f %f %f \n',iter.lambda,delta_f,delta_g);
        % $$$              fclose(fid);
        % $$$         end
        %         if (iter.no_display_fig==0)
        %             figure(20);
        %             imagesc(f_estimated(:,:,128));title('midle slice Simple Gaussian');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/midle_slice_Simple_Gaussian.eps');
        %             figure(17);imagesc(g_chapeau(:,:,sampling.N_phi/2));title('Sinogramme Simple Gaussian');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/Sinogramme_Simple_Gaussian.eps');
        %         end
    case 1
        %         if(iter.save_file ~= 0)
        %             file_name=sprintf('%s/delta_GG.dat',iter.repertoire_test);
        %             fid=fopen(file_name,'a+b');
        %             fprintf(fid,'%f %f %f \n',iter.lambda,delta_f,delta_g);
        %             fclose(fid);
        %         end
        % $$$         if(iter.save_file ~= 0)
        % $$$             file_name=sprintf('%s/delta_GG_beta.dat',iter.repertoire_test);
        % $$$             fid=fopen(file_name,'a+b');
        % $$$             fprintf(fid, ' %f %f %f \n ', iter.beta, delta_f, delta_g);
        % $$$             fclose(fid);
        % $$$         end
        %         if (iter.no_display_fig==0)
        %             figure(21);
        %             imagesc(f_estimated(:,:,128));title('midle slice Generalized Gaussian');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/midle_slice_Generalized_Gaussian.eps');
        %             figure(17);imagesc(g_chapeau(:,:,sampling.N_phi/2));title('Sinogramme Generalized Gaussian');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/Sinogramme_Generalized_Gaussian.eps');
        %         end
    case 2
        % $$$         if(iter.save_file ~= 0)
        % $$$             file_name=sprintf('%s/delta_Cauchy.dat',iter.repertoire_test);
        % $$$             fid=fopen(file_name,'a+b');
        % $$$             fprintf(fid, ' %f %f %f \n ', iter.lambda, delta_f, delta_g);
        % $$$             fclose(fid);
        % $$$         end
        
        %         if (iter.no_display_fig==0)
        %             figure(22);
        %             imagesc(f_estimated(:,:,128));title('midle slice Cauchy');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/midle_slice_Cauchy.eps');
        %             figure(17);imagesc(g_chapeau(:,:,sampling.N_phi/2));title('Sinogramme Cauchy');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/Sinogramme_Cauchy.eps');
        %         end
    case 3
        % $$$         if(iter.save_file ~= 0)
        % $$$             file_name=sprintf('%s/delta_Huber.dat',iter.repertoire_test);
        % $$$             fid=fopen(file_name,'a+b');
        % $$$             fprintf(fid, ' %f %f %f \n ', iter.lambda, delta_f, delta_g);
        % $$$             fclose(fid);
        % $$$         end
        % $$$          if(iter.save_file ~= 0)
        % $$$              file_name=sprintf('%s/delta_Huber_seuil.dat',iter.repertoire_test);
        % $$$              fid=fopen(file_name,'a+b');
        % $$$              fprintf(fid, ' %f %f %f \n ', iter.seuil_huber, delta_f, delta_g);
        % $$$              fclose(fid);
        % $$$          end
        %         if (iter.no_display_fig==0)
        %             figure(23);
        %             imagesc(f_estimated(:,:,128));title('midle slice Huber');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/midle_slice_Huber.eps');
        %             figure(17);imagesc(g_chapeau(:,:,sampling.N_phi/2));title('Sinogramme Huber');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/Sinogramme_Huber.eps');
        %         end
    case 4
        %         if (iter.no_display_fig==0)
        %             figure(24);
        %             imagesc(f_estimated(:,:,128));title('midle slice JMAP');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/midle_slice_JMAP.eps');
        %             figure(17);imagesc(g_chapeau(:,:,sampling.N_phi/2));title('Sinogramme JMAP');colorbar;colormap(gray);drawnow;
        %             print('-depsc','EPS/Sinogramme_JMAP.eps');
        %         end
    otherwise
        disp(['error(methode error when designing middle slice)!!!!' ...
              '!!!!!!!!!!!!!!!'])
        
clear f_real;
clear g_real;
clear iter;        
end


