function [eam_g,eam_relatif_g,eqm_g,eqm_relatif_g]=sauvegarde_projection_TOMO8(g_estimated,g_real,iter)

    s=size(g_real);
    nb_proj=s(3);
    
if (iter.calcul_eqm)
    erreur=g_estimated-g_real;
    eam_g=norm(erreur(:));
    eam_relatif_g=eam_g/norm(g_real(:));
    eqm_g=norm(erreur(:))^2;
    eqm_relatif_g=eqm_g/norm(g_real(:))^2;
else
    eam_g=0;
    eam_relatif_g=0;
    eqm_g=0;
    eqm_relatif_g=0;
end


if (mod(iter.num_iter,iter.save_volume)==0)
    
    %SAUVEGARDE DU PROJECTION EN ENTIER
    if (iter.save_file ~= 0)
        file_name=sprintf('%s/projection_%03d.s',getOutputDirectory(iter),iter.num_iter);
        fid = fopen(file_name, 'wb');
        fwrite(fid,g_estimated ,'float');fclose(fid);
    end
end


if (mod(iter.num_iter,iter.save_file)==0)
    
    %EQM
    if (iter.calcul_eqm)
        file_name=sprintf('%s/eqm_g.dat',getOutputDirectory(iter));
        fid = fopen(file_name, 'a+b');
        fprintf(fid,'%d %f %f\n',iter.num_iter,eqm_g,eqm_relatif_g);fclose(fid);
        file_name=sprintf('%s/eam_g.dat',getOutputDirectory(iter));
        fid = fopen(file_name, 'a+b');
        fprintf(fid,'%d %f %f\n',iter.num_iter,eam_g,eam_relatif_g);fclose(fid);
%         if iter.no_display_fig==0
%             figure(13);plot(eam_relatif_g);title('eam relatif');xlabel('iter');ylabel('eam relatif');drawnow;
%         end
    end
    
    
    %SLICE DU MILIEU
    if (iter.save_file ~= 0)
        file_name=sprintf('%s/middle_slice_g.v',getOutputDirectory(iter));
        fid = fopen(file_name, 'a+b');
        fwrite(fid,g_estimated(:,:,nb_proj/2),'float');
        fclose(fid);
    end
end


end
