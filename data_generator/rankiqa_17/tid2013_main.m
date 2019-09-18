% Generate multiple distortions 
addpath('BM3D')  
% file = dir('./pristine_images/*.bmp');
%file = dir('L:\Datasets\cityscapes_SR-Restore\image\train/*.png');
file = dir('L:\Datasets\IQA\waterloo_samples\*.bmp');

% distortions = [1,2,5,6,7,8,9,10,11,14,15,16,17,18,19,22,23];
% levels = 1:4;
% distortions = [1,6,7,8,15,17,18,23];
distortions = [1,7,8,10,23];
levels = 1:3;
for i = 1:length(file)

    refI = open_bitfield_bmp(fullfile('L:\Datasets\IQA\waterloo_samples', file(i).name));
    for type = distortions    % you could decide which types of distortion you want to generate
        for level = levels        % You could decide how many levels of distortion
            tid2013_generator(refI, type, level,file(i)); 
        end
    end
    fprintf('Finished image %d*16 / %d*16...\n', i,length(file));
    
end






