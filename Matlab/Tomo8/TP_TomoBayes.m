clear all;
N=256;
rep_name=['Projections_',num2str(N),'/'];
iter = Iter3D(rep_name); % Create a class instance




iter.save_file=1;
iter.save_volume=2;
iter.fc=1.0;%frequence de coupure pour filtre de rétroporjection filtrée


f_real=CreateVolumeReal(iter);
g_real=getSinoReal(iter);

[g_real,rsb_in]=addNoise(iter,g_real);
file_name=sprintf('%s/P_ER_GPU_NOISE_%2.1fdB.s',iter.workdirectory,rsb_in);
fid = fopen(file_name, 'wb');
fwrite(fid,g_real ,'float');
fclose(fid);


    
% RECONSTRUCTION RETROPROJECTION FILTREE
 %f_FDK=doFDK(iter,g_real,iter.fc);
 %file_name=sprintf('%s/volume_FDK.v',getOutputDirectory(iter));
 %fid = fopen(file_name, 'wb');
 %fwrite(fid,f_FDK ,'float');
 %fclose(fid);
        
       
% RECONSTRUCTION ITERATIVE MOINDRE CARRE AVEC REGULARISATION QUADRATIQUE
%[f_estimated,iter]=gradient_GCH_TOMO8(f_estimated,f_real, g_real,iter);
%setPositivity(iter,0);
setLambda(iter,1.0);
f_estimated_lambda_1=CreateVolumeInit(iter);
doGradient(iter,f_estimated_lambda_1,g_real,f_real);
 
 setLambda(iter,0.1);
f_estimated_lambda_0_1=CreateVolumeInit(iter);
 doGradient(iter,f_estimated_lambda_0_1,g_real,f_real);
 
 setLambda(iter,0);
f_estimated_lambda_0=CreateVolumeInit(iter);
 doGradient(iter,f_estimated_lambda_0,g_real,f_real);
 
 
 figure(1);
 %figure(1);plot(f_FDK(:,N/2,N/2),'r','LineWidth',1.5,'Marker','o');hold on;
 figure(1);plot(f_estimated_lambda_1(:,N/2,N/2),'b','LineWidth',1.5,'Marker','+');hold on;
 figure(1);plot(f_estimated_lambda_0_1(:,N/2,N/2),'g','LineWidth',1.5,'Marker','*');hold on;
 plot(f_estimated_lambda_0(:,N/2,N/2),'m','LineWidth',1.5,'Marker','x');hold on;
 figure(1);plot(f_real(:,N/2,N/2),'k','LineWidth',1.5);
 legend('Iteratif lambda=1','Iteratif lambda=0.1','Iteratif lambda=0'...
  ,'real');
 

  figure(2);
  imagesc(f_real(:,:,N/2));title('midle slice real');colorbar;colormap(gray);drawnow;
 figure(3);
  imagesc(f_estimated_lambda_0_01(:,:,N/2));title('midle slice lambda 0.01');colorbar;colormap(gray);drawnow;
 figure(4);
  imagesc(f_estimated_lambda_0(:,:,N/2));title('midle slice lambda 0');colorbar;colormap(gray);drawnow;
 
 
%%

%f_chapeau=f_estimated;
%g_chapeau=single(zeros(sampling.N_un,sampling.N_vn,sampling.N_phi));
%projection3D_TOMOX_mex(f_chapeau,g_chapeau,kernel_P,constante_GPU, sampling,iter.projecteur);

%delta_f=((sum((f_chapeau(:)-f_real(:)).^2)))/((sum((f_real(:)).^2)));
%delta_g=((sum((g_chapeau(:)-g_real(:)).^2)))/((sum((g_real(:)).^2)));
%iter.lambda

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
   !!!!!!!!!!'])
        