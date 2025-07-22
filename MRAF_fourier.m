clear;close all;clc;
%% variable parameter
m=1080;n=1080;
mm=2160;nn=2160;
ob_str='woodpiles.bmp';
hn_str='hologram_woodpiles_mraf.bmp';
obj=mat2gray(imread(ob_str));
lamda=8e-7;                      
pix=8e-6;  
t0=50;
w=470;
p=0.5;
alpha=9e-3;
%% fixed parameter
k=2*pi/lamda;  
%% signal region
obj_sg=remove_padding(obj,m-2*w,n-2*w)-0.2;
obj_sg=zero_padding(obj_sg,m,n)+0.2;
%% generate initial phase
[x,y]=meshgrid(linspace(-n*pix/2,n*pix/2,n),linspace(-m/2*pix,m/2*pix,m));
phi_in=k*sqrt(x.^2+y.^2).*tan(alpha);
phi_in=angle(exp(1i*phi_in));
figure;imshow(phi_in,[])
%% MRAF algorithm
tic
for t=1:t0
if t==1
compholo=exp(1i*phi_in);
else
compholo=exp(1i*phi);
end
u2=fftshift(fft2(fftshift(compholo))); 
img_phi=angle(u2);
img=abs(u2);
img_1=mat2gray(abs(u2));
for c=1:m
    for d=1:n
         if w+1<=d&&d<=n-w&&w+1<=c&&c<=m-w
            amp(c,d)=p*obj_sg(c,d)+(1-p)*p*img_1(c,d);
         else
            amp(c,d)=img_1(c,d);
         end
    end 
end
if t~=t0
obj_com=amp.*exp(1i*img_phi);
U2=fftshift(ifft2(fftshift(obj_com)));
phi=angle(U2);
POH=mat2gray(phi);
end
end
toc
%% data display
figure;
subplot(1,2,1);imshow(POH, []);title('Hologram');
subplot(1,2,2);imshow(img, []);title('I');
%% data storage
imwrite(POH,hn_str)