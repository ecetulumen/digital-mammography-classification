%% Objective image-quality evaluation for noise-filter pairs
% Compares salt-and-pepper and Gaussian noise after median, Gaussian, and
% Wiener filtering. BRISQUE, PIQE, and NIQE are no-reference metrics where
% lower scores indicate better perceived image quality.

clc;
clear;
close all;
rng(42, 'twister');

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
image_folder = fullfile(project_root, 'data', 'metric_images');
output_folder = fullfile(project_root, 'results', 'filter_selection');

if ~isfolder(image_folder)
    error('Input folder not found: %s', image_folder);
end
if ~isfolder(output_folder)
    mkdir(output_folder);
end

patterns = {'*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'};
image_files = dir(fullfile(image_folder, patterns{1}));
for pattern_index = 2:numel(patterns)
    image_files = [image_files; dir(fullfile(image_folder, patterns{pattern_index}))]; %#ok<AGROW>
end
if isempty(image_files)
    error('No supported images found in: %s', image_folder);
end

variant_names = [
    "original"
    "salt_pepper_noisy"
    "salt_pepper_median"
    "salt_pepper_gaussian"
    "salt_pepper_wiener"
    "gaussian_noisy"
    "gaussian_median"
    "gaussian_gaussian"
    "gaussian_wiener"
];
noise_names = [
    "none"
    "salt_pepper"
    "salt_pepper"
    "salt_pepper"
    "salt_pepper"
    "gaussian"
    "gaussian"
    "gaussian"
    "gaussian"
];
filter_names = [
    "none"
    "none"
    "median"
    "gaussian"
    "wiener"
    "none"
    "median"
    "gaussian"
    "wiener"
];

row_count = numel(image_files) * numel(variant_names);
image_column = strings(row_count, 1);
variant_column = strings(row_count, 1);
noise_column = strings(row_count, 1);
filter_column = strings(row_count, 1);
brisque_column = zeros(row_count, 1);
piqe_column = zeros(row_count, 1);
niqe_column = zeros(row_count, 1);

row = 0;
for image_index = 1:numel(image_files)
    image_path = fullfile(image_folder, image_files(image_index).name);
    image = imread(image_path);
    if size(image, 3) == 3
        image = rgb2gray(image);
    end

    salt_pepper_noisy = imnoise(image, 'salt & pepper', 0.02);
    gaussian_noisy = imnoise(image, 'gaussian', 0, 0.02);

    variants = {
        image
        salt_pepper_noisy
        medfilt2(salt_pepper_noisy, [3 3])
        imgaussfilt(salt_pepper_noisy, 2)
        wiener2(salt_pepper_noisy, [5 5])
        gaussian_noisy
        medfilt2(gaussian_noisy, [3 3])
        imgaussfilt(gaussian_noisy, 2)
        wiener2(gaussian_noisy, [5 5])
    };

    for variant_index = 1:numel(variants)
        row = row + 1;
        candidate = variants{variant_index};
        image_column(row) = string(image_files(image_index).name);
        variant_column(row) = variant_names(variant_index);
        noise_column(row) = noise_names(variant_index);
        filter_column(row) = filter_names(variant_index);
        brisque_column(row) = brisque(candidate);
        piqe_column(row) = piqe(candidate);
        niqe_column(row) = niqe(candidate);
    end
end

per_image = table( ...
    image_column, variant_column, noise_column, filter_column, ...
    brisque_column, piqe_column, niqe_column, ...
    'VariableNames', {'Image', 'Variant', 'Noise', 'Filter', 'BRISQUE', 'PIQE', 'NIQE'} ...
);
writetable(per_image, fullfile(output_folder, 'filter_metrics_per_image.csv'));

[groups, variants, noises, filters] = findgroups( ...
    per_image.Variant, per_image.Noise, per_image.Filter ...
);
summary = table( ...
    variants, noises, filters, ...
    splitapply(@mean, per_image.BRISQUE, groups), ...
    splitapply(@std, per_image.BRISQUE, groups), ...
    splitapply(@mean, per_image.PIQE, groups), ...
    splitapply(@std, per_image.PIQE, groups), ...
    splitapply(@mean, per_image.NIQE, groups), ...
    splitapply(@std, per_image.NIQE, groups), ...
    'VariableNames', {
        'Variant', 'Noise', 'Filter', ...
        'MeanBRISQUE', 'StdBRISQUE', ...
        'MeanPIQE', 'StdPIQE', ...
        'MeanNIQE', 'StdNIQE'
    } ...
);
writetable(summary, fullfile(output_folder, 'filter_metrics_summary.csv'));

ranking = summary(summary.Filter ~= "none", :);
ranking.NormalizedScore = ...
    minmax_score(ranking.MeanBRISQUE) + ...
    minmax_score(ranking.MeanPIQE) + ...
    minmax_score(ranking.MeanNIQE);
ranking = sortrows(ranking, 'NormalizedScore', 'ascend');
writetable(ranking, fullfile(output_folder, 'filter_pair_ranking.csv'));

disp('Filter-pair ranking (lower is better):');
disp(ranking(:, {'Noise', 'Filter', 'MeanBRISQUE', 'MeanPIQE', 'MeanNIQE', 'NormalizedScore'}));
fprintf('Best pair: %s noise + %s filter\n', ranking.Noise(1), ranking.Filter(1));


function score = minmax_score(values)
    minimum = min(values);
    maximum = max(values);
    if maximum == minimum
        score = zeros(size(values));
    else
        score = (values - minimum) ./ (maximum - minimum);
    end
end
