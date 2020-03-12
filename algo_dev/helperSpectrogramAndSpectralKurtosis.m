function helperSpectrogramAndSpectralKurtosis(x, fs, level)
%HELPERSPECTROGRAMANDSPECTRALKURTOSIS Compute and plot spectrogram and
%spectral kurtosis. Spectral kurtosis is plotted on the side of
%the spectrogram.

% Copyright 2017 The MathWorks, Inc.
[~, ~, ~, fc, wc, BW] = kurtogram(x, fs, level);
[sk, fsk] = pkurtosis(x, fs, wc);

figure
subplot('position', [0.35 0.12 0.6 0.82])
pspectrum(x, fs, 'spectrogram', 'Reassign', true);
title('')
hold on
xlimits1 = get(gca, 'XLim');
ylimits1 = get(gca, 'YLim');
scale = ylimits1(2)/(fs/2);
fc_scaled = fc*scale;
BW_scaled = BW*scale;
xlim(xlimits1)
plot(xlimits1, [fc_scaled fc_scaled])
plot([xlimits1 NaN xlimits1], [fc_scaled-BW_scaled/2 fc_scaled-BW_scaled/2 NaN fc_scaled+BW_scaled/2 fc_scaled+BW_scaled/2], '--')
hold off

subplot('position', [0.05 0.12 0.15 0.82])
plot(fsk*scale, sk)
hold on
ylimits2 = [min(sk(:)) - 0.5, max(sk(:)) + 0.5];
xlim(ylimits1)
ylim(ylimits2)
plot([fc_scaled fc_scaled], ylimits2)
plot([fc_scaled-BW_scaled/2 fc_scaled-BW_scaled/2 NaN fc_scaled+BW_scaled/2 fc_scaled+BW_scaled/2], [ylimits2 NaN ylimits2], '--')
ylabel('Spectral Kurtosis')
camroll(90)
hold off
end