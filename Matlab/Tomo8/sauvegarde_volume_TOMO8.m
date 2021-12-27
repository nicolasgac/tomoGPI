function [iter_out]=sauvegarde_volume_TOMO8(f_estimated,f_real,iter)

    s=size(f_real);
    nb_slice=s(3);
    
if (iter.calcul_eqm)
    erreur=f_estimated-f_real;
    iter.eam_f(iter.num_iter)=norm(erreur(:));
    iter.eam_relatif_f(iter.num_iter)=iter.eam_f(iter.num_iter)/norm(f_real(:));
    iter.eqm_f(iter.num_iter)=norm(erreur(:))^2;
    iter.eqm_relatif_f(iter.num_iter)=iter.eqm_f(iter.num_iter)/norm(f_real(:))^2;
end


if (mod(iter.num_iter,iter.save_volume)==0)
    
    %SAUVEGARDE DU VOLUME EN ENTIER
    if (iter.save_file ~= 0)
        file_name=sprintf('%s/volume_%03d.v',getOutputDirectory(iter),iter.num_iter);
        fid = fopen(file_name, 'wb');
        fwrite(fid,f_estimated ,'float');
        fclose(fid);
    end
end


if (mod(iter.num_iter,iter.save_file)==0)
    
    %EQM
    if (iter.calcul_eqm)
        file_name=sprintf('%s/eqm_f.dat',getOutputDirectory(iter));
        fid = fopen(file_name, 'a+b');
        fprintf(fid,'%d %f %f\n',iter.num_iter,iter.eqm_f(iter.num_iter),iter.eqm_relatif_f(iter.num_iter));fclose(fid);
        file_name=sprintf('%s/eam_f.dat',getOutputDirectory(iter));
        fid = fopen(file_name, 'a+b');
        fprintf(fid,'%d %f %f\n',iter.num_iter,iter.eam_f(iter.num_iter),iter.eam_relatif_f(iter.num_iter));fclose(fid);
%         if iter.no_display_fig==0
%             figure(13);plot(iter.eam_relatif_f);title('eam relatif');xlabel('iter');ylabel('eam relatif');drawnow;
%         end
    end
    
    
    %SLICE DU MILIEU
    if (iter.save_file ~= 0)
        file_name=sprintf('%s/middle_slice_f.v',getOutputDirectory(iter));
        fid = fopen(file_name, 'a+b');
        fwrite(fid,f_estimated(:,:,nb_slice/2),'float');
        fclose(fid);
    end
end



if iter.no_display_fig==0
    figure(2);imagesc(f_estimated(:,:,nb_slice/2));title(['Image reconstruite (Iter global=',num2str(iter.num_iter),')']);colorbar;drawnow;
end

iter_out=iter;
end
