clear all;
close all;
iter = Iter3D('Projections_256/'); % Create a class instance




iter.save_file=1;
iter.save_volume=2;
iter.fc=1.0;%frequence de coupure pour filtre de rétroporjection filtrée




f_real=CreateVolumeReal(iter);
g_real=getSinoReal(iter);

rsb_in=20;
[g_real,rsb_in,b]=addNoise(iter,g_real);
file_name=sprintf('%s/P_ER_GPU_NOISE_%2.1fdB.s',iter.workdirectory,rsb_in);
fid = fopen(file_name, 'wb');
fwrite(fid,g_real ,'float');
fclose(fid);

 
% RECONSTRUCTION RETROPROJECTION FILTREE
f_FDK=doFDK(iter,g_real,iter.fc);
file_name=sprintf('%s/volume_FDK.v',getOutputDirectory(iter));
fid = fopen(file_name, 'wb');
fwrite(fid,f_FDK ,'float');
fclose(fid);

iter.fc=0.2;
f_FDK_bf=doFDK(iter,g_real,iter.fc);
file_name=sprintf('%s/volume_FDK_filtre_0_2.v',getOutputDirectory(iter));
fid = fopen(file_name, 'wb');
fwrite(fid,f_FDK_bf ,'float');
fclose(fid);

      setLambda(iter,0);
     f_estimated_lambda=f_FDK_bf;
 doGradient(iter,f_estimated_lambda,g_real,f_real);
  diff=(f_estimated_lambda-f_real).^2;
 err_lambda0=sum(diff(:))/sum(f_real(:).^2);
 diff=(f_FDK-f_real).^2;
 err_FDK=sum(diff(:))/sum(f_real(:).^2);
     figure(1);plot(f_real(:,128,128),'LineWidth',1.5);hold on;
 figure(1);plot(f_FDK(:,128,128),'LineWidth',1.5,'Marker','o');hold on;
 figure(1);plot(f_estimated_lambda(:,128,128),'LineWidth',1.5,'Marker','o');hold on;
     
%% RECONSTRUCTION ITERATIVE MOINDRE CARRE AVEC REGULARISATION QUADRATIQUE
%[f_estimated,iter]=gradient_GCH_TOMO8(f_estimated,f_real, g_real,iter);
setPositivity(iter,0);
%color=['y','g','r','y','g','r','y','g','r','y','g','r'];

%setGradientIterationNb(50);
for i=-5 : 5
    lambda(i+6)=10^i;
    setLambda(iter,lambda(i+6));
f_estimated_lambda=f_FDK_bf;%CreateVolumeInit(iter);
 doGradient(iter,f_estimated_lambda,g_real,f_real);
 figure(1);plot(f_estimated_lambda(:,128,128),'LineWidth',1.5,'Marker','+');hold on;
    diff=(f_estimated_lambda-f_real).^2;
    
    err(i+6)=sum(diff(:))/sum(f_real(:).^2);
   
end
 
 legend('real','fbp','lambda=0',['log(lambda)=' int2str(-5)],['log(lambda)=' int2str(-4)],['log(lambda)=' int2str(-3)],['log(lambda)=' int2str(-2)],['log(lambda)=' int2str(-1)],['log(lambda)=' int2str(0)],['log(lambda)=' int2str(1)],['log(lambda)=' int2str(2)],['log(lambda)=' int2str(3)],['log(lambda)=' int2str(4)],['log(lambda)=' int2str(5)]);

 
 
 
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
        
