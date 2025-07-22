clear;close all;clc;
%% variable parameter
m=1080;n=1080;
mm=2160;nn=2160;  %after zero padding
ob_str='woodpiles.bmp';
hn_str='hologram_woodpiles_gs.bmp';
obj=imread(ob_str);
obj=mat2gray(imresize(im2gray(obj),[m,n]));
z=0.16;
lamda=8e-7;       
pix=8e-6;
t0=50;    %times of iteration
%% fixed parameter
k=2*pi/lamda;  
[fx,fy]=meshgrid(linspace(-1/(2*pix),1/(2*pix),nn),linspace(-1/(2*pix),1/(2*pix),mm));
%% GS algorithm
for t=1:t0
    if t==1
        obj_com=obj.*exp(1i*2*pi*rand(m,n));
    else
        obj_com=obj.*exp(1i*img_phi);
    end
  obj_com_0=zero_padding(obj_com,mm,nn);
  U1_0=fftshift(fft2(fftshift(obj_com_0))); 
  H_AS_0=exp(1i*k*z.*sqrt(1-(lamda*fx).^2-(lamda*fy).^2));
  U2_0=fftshift(ifft2(fftshift(U1_0.*H_AS_0))); 
  U2=remove_padding(U2_0,m,n);
  phi=angle(U2);
%% reconstruction
  compholo=exp(1i*phi);    
  compholo_0=zero_padding(compholo,mm,nn); 
  u1_0=fftshift(fft2(fftshift(compholo_0))); 
  h_AS_0=exp(1i*k*(-z).*sqrt(1-(lamda*fx).^2-(lamda*fy).^2));
  u2_0=fftshift(ifft2(fftshift(u1_0.*h_AS_0))); 
  u2=remove_padding(u2_0,m,n);
  img=abs(u2);   
  img_phi=angle(u2);
end
%% data display
figure;
subplot(1,2,1);imshow(mat2gray(phi), []);title('Hologram');
subplot(1,2,2);imshow(mat2gray(img), []);title('I');
set(gcf, 'Position', [100 100 1200 1200]);
%% data storage
POH=mat2gray(phi);imwrite(POH,hn_str)