function [sinogram_filtered]=ramp_filter_TOMO8(sinogram,sur_ech,fc_FDK,delta_un);
%
%
%
%
%
%
%
%
%
%
%
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CREATION DE LA RAMPE DE TAILLE sur_ech*N
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


N_un=size(sinogram,1);
N_vn=size(sinogram,2);
N_phi=size(sinogram,3);

N_un_sur_ech=N_un*sur_ech;

for un = 1 : sur_ech/2*N_un+1
    FFT_ramp(un)=-(un-1)+sur_ech/2*N_un;
end

for un = sur_ech/2*N_un+2 : sur_ech*N_un
    FFT_ramp(un)=(un-1)-sur_ech/2*N_un;
end


%figure(10);plot(FFT_ramp(:));drawnow;

FFT_ramp=fftshift(FFT_ramp);
FFT_ramp=FFT_ramp/max(FFT_ramp);%0.5* NG juin 2015

%figure(11);plot(FFT_ramp(:));drawnow;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FENETRAGE DU FILTRE RAMPE AVEC HANNING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_ramp=round(N_un_sur_ech*fc_FDK);
%N_ramp
%mod(N_ramp,2)

if (mod(N_ramp,2)==1)
  N_ramp=N_ramp-1;
end

%N_ramp

hanning_filter_center=hann(N_ramp)';
%size(hanning_filter_center)
%figure(1);plot(hanning_filter_center(:));drawnow;

  hanning_filter=[zeros(1,(N_un_sur_ech-N_ramp)/2),hanning_filter_center,zeros(1,(N_un_sur_ech-N_ramp)/2)];


size(hanning_filter)
%figure(2);plot(hanning_filter(:));drawnow;


%size(hanning_filter_center,2)+N_un_sur_ech*(1-fc_FDK)



hanning_filter=fftshift(hanning_filter);

%figure(20);plot(hanning_filter(:));drawnow;
FFT_ramp_hanning=FFT_ramp.*hanning_filter;



%figure(4);plot(FFT_ramp_hanning(:));drawnow;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILTRE RAMPE SUR 2*N
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ramp_hanning=real(ifft(FFT_ramp_hanning));


%figure(5);plot(ramp_hanning(:));drawnow;

ramp_hanning_downscale=[ramp_hanning(1:round(N_un)), ramp_hanning(N_un_sur_ech-round(N_un)+1:N_un_sur_ech)];

ramp_hanning_downscale(round(N_un/2)+1:round(3*N_un/2))=0;


%figure(6);plot(ramp_hanning_downscale(:));drawnow;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLICATION DU FILTRE RAMPE DANS FOURIER (AVEC UN ZERO PADDING SUR LE SINOGRAMME)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



FFT_filter= fft(ramp_hanning_downscale).';

%figure(7);plot(real(fft(ramp_hanning_downscale)));drawnow;

%%% a enlever 
%FFT_filter_ref=  [0:N_un,N_un-1:-1:1 ]'; 


%keyboard;

for phi = 1 : N_phi

    for vn = 1 : N_vn	
	temp=ifft(fft(sinogram(:,vn,phi),2*N_un).*FFT_filter);
     	%temp=ifft(real(fft(sinogram(:,vn,phi),2*N_un)).*real(FFT_filter));
	sinogram_filtered(:,vn,phi)=real(temp(1:N_un))./(2*delta_un);
	   

    end	


end
figure(41);imagesc(sinogram_filtered(:,:,1));drawnow;
