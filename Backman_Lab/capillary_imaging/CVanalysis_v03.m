% Finds edges of capillary networks in images, computes area of vessels by clustering, in order to examine if capillary network structure can distinguish between healthy and cancerous tissue.
%Read in your data....
%clear all

cd ('G:\EIBS CellviZio\BMP Images');
directory = cd;

mousefile = 'xx83344'; %'xx83344*';
ext = '*.bmp';
filesim=dir(strcat(mousefile, ext)); %place all images into a filelist
% mpix = 427; 
% npix = 427; 

values =[];
Vessel_param = {};

% set threshold values:
uppert = 0.33;
lowert = 0.25;

gaussianFilter = fspecial('gaussian', [2 5], 50); % Guassian filter
% Determine canny operator to use in analysis 
cannyop=1;  % 1=basic, 2=advanced 

% Pixel conversion to microns:
% original files note 512 pixels for 239 microns
pixel = 512/239; % conversion 2.14pixels/1micron
%% Image Conversions for Future Filtering
 %images= zeros(mpix, npix, size(filesim,1)); %initialize array for data storage
images=[];

for j = 1:size(filesim,1)
    frame = filesim(j).name((end-7):(end-4));
    mouseID = filesim(j).name(3:7);
    nickname = [filesim(j).name(3:11) '-' frame];
    movie = filesim(j).name((end-13):(end-10));
    data = imread(filesim(j).name);
    ypix = size(data,1); % #pixels in x directiion
    xpix = size(data,2); % #pixels in y directiion
    % Create the circle mask for each image: (a logical image of a circle
    % with specified diameter, center, and image size)
    % First create the image.
    [columnsInImage rowsInImage] = meshgrid(1:xpix, 1:ypix);
    % Next create the circle in the image.
    if mod(xpix,2) == 1
        centerX = (xpix+1)/2;        
    else
        centerX = xpix/2;
    end
    if mod(ypix,2) ==1
        centerY = (ypix+1)/2;
    else
        centerY = ypix/2;
    end
    if mod(max(size(data)),2) == 1
        radius = (max(size(data))-1)/2;
    else
        radius = max(size(data))/2;
    end
    circlePixels = sqrt((rowsInImage - centerY).^2 + (columnsInImage - centerX).^2) <= radius;
    % circlePixels is a 2D "logical" array.
     
    % Convert data and apply circlePixels mask:
    images(:,:,j) = rgb2gray(data);
    Idouble(:,:,j) = double(images(:,:,j)); %convert u8 data into double format
    Imask(:,:,j) =  Idouble(:,:,j).*circlePixels; %Apply the circle mask and label mask that have been previously defined
    
    % Remove Noise via Wiener Filter
    Wfilt15(:,:,j) = wiener2(Imask(:,:,j),[15 15]); 
    Ione(:,:,j) = Wfilt15(:,:,j)./255; % Scale values to range from 0 to 1
    
    % Contrast Enhancement operator
    Imask_adapthisteq = adapthisteq(Ione(:,:,j),'NumTiles',[15 15],'ClipLimit',0.005,'NBins',256,'Range','original','Distribution','uniform');
    Icontrast(:,:,j) = Imask_adapthisteq;    
    
    % Gaussian Lowpass Filter Applied to Masked Image for Smoothing (filter defined above)
    gI(:,:,j) = imfilter(Wfilt15(:,:,j), gaussianFilter, 'symmetric', 'conv');
    gIone(:,:,j) = gI(:,:,j)./255; % convert guassian filtered image to scale [0 1]
    
    Ithresh = zeros(mpix,npix); % Thresholding image for binary result
    Ithresh(Imask_adapthisteq>uppert)=1;
    Ithresh(Imask_adapthisteq<=lowert)=0;
%     Ithresh(gIone(:,:,j)>uppert)=1;
%     Ithresh(gIone(:,:,j)<=lowert)=0;
    Ibinary(:,:,j) = Ithresh;
    
    if cannyop==1; %apply canny operator to image after noise filetr and circle mask
        bwcannyIm(:,:,j) = edge(gI(:,:,j), 'canny', 0.135, 3); 
    elseif cannyop==2; 
        %Advanced Canny Filter 
        %Parameters of edge detecting filters: 
        Nx1=20;Sigmax1=75;Nx2=20;Sigmax2=60;Theta1=pi/4;  % X-axis direction filter
        Ny1=20;Sigmay1=5;Ny2=10;Sigmay2=70;Theta2=0;  % Y-axis direction filter
        alfa=0.035;  % The thresholding parameter alfa:
        % X-axis direction edge detection 
        filterx = d2dgauss(Nx1,Sigmax1,Nx2,Sigmax2,Theta1);  
        Ix(:,:,j) = conv2(gI(:,:,j),filterx,'same');
        % Y-axis direction edge detection
        filtery = d2dgauss(Ny1,Sigmay1,Ny2,Sigmay2,Theta2);  
        Iy(:,:,j) = conv2(gI(:,:,j),filtery,'same'); 
        % Norm of the gradient (Combining the X and Y directional derivatives)
        NVI(:,:,j) = sqrt(Ix(:,:,j).*Ix(:,:,j)+Iy(:,:,j).*Iy(:,:,j)); 
        % Thresholding
        I_max(j)=max(max(NVI(:,:,j)));
        I_min(j)=min(min(NVI(:,:,j)));
        level(j)=alfa*(I_max(j)-I_min(j))+I_min(j);
        Ibw(:,:,j) = max(NVI(:,:,j),level.*ones(size(NVI)));
        % Thinning (Using interpolation to find the pixels where the norms of gradient are local maximum.)
        [n,m]=size(Ibw(:,:,j));
        for k=2:n-1,
        for l=2:m-1,
            if Ibw(k,l,j) > level,
               X=[-1,0,+1;-1,0,+1;-1,0,+1];
               Y=[-1,-1,-1;0,0,0;+1,+1,+1];
               Z=[Ibw(k-1,l-1,j),Ibw(k-1,l,j),Ibw(k-1,l+1,j);
                  Ibw(k,l-1,j),Ibw(k,l,j),Ibw(k,l+1,j);
                  Ibw(k+1,l-1,j),Ibw(k+1,l,j),Ibw(k+1,l+1,j)];
               XI=[Ix(k,l,j)/NVI(k,l,j), -Ix(k,l,j)/NVI(k,l,j)];
               YI=[Iy(k,l,j)/NVI(k,l,j), -Iy(k,l,j)/NVI(k,l,j)];
               ZI=interp2(X,Y,Z,XI,YI);
               if Ibw(k,l,j) >= ZI(1) & Ibw(k,l,j) >= ZI(2)
                   I_temp(k,l)=I_max(j);
               else
                   I_temp(k,l)=I_min(j);
               end
            else 
                I_temp(k,l)=I_min(j);
            end
            bwcannyIm(:,:,j) = I_temp;
        end
        end
    end
    %  Overlay Images to evaluate vessel outline defined by canny operator:
    out_r = gIone(:,:,j);
    out_g = gIone(:,:,j);
    out_b = gIone(:,:,j);
    
    out_r(bwcannyIm(:,:,j)) = 1;
    out_g(bwcannyIm(:,:,j)) = 0;
    out_b(bwcannyIm(:,:,j)) = 0;
    
%     out_r(themask) = 0;
%     out_g(themask) = .2;
%     out_b(themask) = .4; 
%     
    outall = cat(3, out_r, out_g, out_b);
    out(:,:,:,j) = outall;
    % End of Image Analysis
    %% Begin Quantitative analysis:
    % Calculate Microvessel Density in Image Circle
        totarea = sum(sum(circlePixels));  %calculate total # pixels in tmask
        vessels = sum(sum(Ibinary(:,:,j),1));  %Calculate total # pixels = vessels in image:
        mvd = vessels/totarea;
    % Calculate Fourier of images (original, filtered, binary)
    
    values ={filesim(j).name,nickname,mouseID,movie,frame,vessels,totarea,mvd}; %values calculated for each image
    Vessel_param = [Vessel_param; values];
    
    %% Plot all images in one panel
    figure,  
    subplot(2,3,1);
    imagesc(Idouble(:,:,j)), colormap(gray);
    title([nickname]);axis off;
    
    subplot(2,3,2);
    imagesc(Wfilt15(:,:,j)), colormap(gray);
    title('Wiener Filter: Noise Reduction');axis off;
    
    subplot(2,3,3);
    imagesc(Icontrast(:,:,j)), colormap(gray);
    title('Contrast Enhancement');axis off;
    
    subplot(2,3,4);
    imagesc(Ibinary(:,:,j)), colormap(gray);
    title('Binary Image of Vessels');axis off;
    
    subplot(2,3,5);
    imagesc(bwcannyIm(:,:,j)), colormap(gray);
    title('Vessel Segmentation');axis off;
    
    subplot(2,3,6);
    imagesc(out(:,:,:,j)); 
    title('Check Vessel Outline');axis off;
    
    makeprettyCV;

%    % Save Image Panel as Figures
%    saveas(gcf, filesim(j).name(1:(end-4)),'fig')
%    saveas(gcf, filesim(j).name(1:(end-4)),'emf')
%    saveas(gcf, filesim(j).name(1:(end-4)),'tif')
%     
end

%% Create Output File with Image Parameters
% 
% zcolumn_header = {'filename','nickname','mouseid','movie','frameid','vessels','totalarea','mvd'};
% final= [zcolumn_header;Vessel_param];
% xlswrite('EIBS_CV_imageanal01.xls',final,'E1 nums 4');

%% Fourier transform:
j=1
frame = filesim(j).name((end-7):(end-4));
mouseID = filesim(j).name(3:7);
nickname = [filesim(j).name(3:11) '-' frame];

% Idb = abs(fft2(Idouble(:,:,j)));  %ACF
% Idouble_fft = fftshift(ifft2(Idb));
clear images; clear Idouble; clear Imask; 
for j = 1:size(filesim,1)
    frame = filesim(j).name((end-7):(end-4));
    mouseID = filesim(j).name(3:7);
    nickname = [filesim(j).name(3:11) '-' frame];
    movie = filesim(j).name((end-13):(end-10));
    data = imread(filesim(j).name);
    mpix = size(data,1); % #pixels in x directiion
    npix = size(data,2); % #pixels in y directiion
    images(:,:,j) = rgb2gray(data);
    Idouble(:,:,j) = double(images(:,:,j));
    Imask(:,:,j) =  Idouble(:,:,j).*cvmask; 


    Idb2 = fftshift(abs(fft2(Imask(:,:,j))));  
    rotA(:,j) = rotavg(Idb2,floor(size(Idb2,1)/2-1),floor(size(Idb2,2)/2+1),floor(size(Idb2,1)/2+1));
end

%Ire2 = fftshift(real(fft2(Idouble(:,:,j))));
%Idb_C = Imask;
%I = [Idb_orig
%normalize each FFT image to the max value in that image
for j = 1:4
    rotC(:,j) = rotC(:,j)/max(rotC(:,j));
    rotA(:,j) = rotA(:,j)/max(rotA(:,j));
end

xaxC = (1:mpix/2)*pixel; %.5621; %pixel;

% figure, plot(xax, rot); title([nickname]);
figure, semilogy(xaxC,rotC(:,1),'b', xax,rotA(:,1),'r', xaxC, rotC(:,2:3),'b', xax,rotA(:,2:3),'r')
xlabel('Spatial Frequency (au)'), legend('Control','AOM');
makepretty
%% ACF
% % IWf = abs(fft2(Wfilt15(:,:,j))).^2;
% % Wfilt15_fft = fftshift(ifft2(IWf));
% % 
% % Icn = abs(fft2(Icontrast(:,:,j))).^2;
% % Icontrast_fft = fftshift(ifft2(Icn));
% % 
% % Ibn = abs(fft2(Ibinary(:,:,j))).^2;
% % Ibinary_fft = fftshift(ifft2(Ibn));

%% Create circle mask

% Create a logical image of a circle with specified
% diameter, center, and image size.
% First create the image.
imageSizeX = size(data,2);
imageSizeY = size(A2,1);
[columnsInImage rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
% Next create the circle in the image.
centerX = imageSizeX/2;
centerY = imageSizeY/2;
radius = max(size(A2))/2; %190;
circlePixels = sqrt((rowsInImage - centerY).^2 + (columnsInImage - centerX).^2) <= radius;
% circlePixels is a 2D "logical" array.
% Now, display it.
figure;
image(circlePixels);colormap([0 0 0; 1 1 1]);
title('Binary image of a circle');
%% Skeletonized vessel outline
bwskel = Ibinary;
for j = 1:4
    BW2(:,:,j) = bwmorph(bwskel(:,:,j),'remove');
    BW3 (:,:,j) = bwmorph(bwskel(:,:,j),'skel',Inf);
end
figure, 
subplot 221, imagesc(BW2(:,:,1)), colormap(gray), title('remove')
subplot 222, imagesc(BW3(:,:,1)), colormap(gray), title('skel')
%% Create window mask for files less than 512 x 512
newframe = zeros([512 512]);
widx = (512-(size(Idouble,1)+1))/2;
ht = (512-(size(Idouble,2)+1))/2;
maskframe = zeros(mpix, npix); maskframe2 = 1 + maskframe;

%horizontal concatenate for mask creation
A1 = horzcat(maskframe, maskframe2);
A1 = A1(:,(size(data,2)-widx):end);
maskframe = zeros([size(A1,1) size(A1,2)]);
A1 = horzcat(A1, maskframe);
mask_sides = A1(:,1:(size(data,2)+(2*widx)+1));
figure, imagesc(mask_sides); size(A1)
%vertical concatenate for mask creation
maskframe = zeros([size(A1,1) size(A1,2)]);
A2 = vertcat(maskframe, A1);
A2 = A2((size(data,2)-ht):end,:);
maskframe = zeros([size(A2,1) size(A2,2)]);
A2 = vertcat(A2,maskframe);
A2 = A2(1:(size(data,2)+(2*ht+1)),:);
figure, imagesc(A2); size(A2)

% try1 = 0:widx;
% cat(1, newframe(0:widx, 0:ht) maskframe(:,:)): newframe((end-widx):end),(end-ht):end);
