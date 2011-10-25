% select good features using correlation feature selection

function [good] = select_features(sample, step, n)

	assert(n > 0 & n < size(sample, 2));

	d = size(sample, 2);
	score = -Inf;
	best = [];

	ffCorr = corr(sample); % inter-feature
	cfCorr = zeros(1, d); % feature-class

% add an ordinal denoting class to each sample
	for i = 1:size(sample, 1)
		sample(i, d+1) = ceil(i/step);
	end

	for i = 1:d
		cFCorr(i) = corr(sample(:, i), sample(:, d+1));
	end

% enumerate and evaluate the d choose n combinations of features
	combs = nchoosek(1:d, n);
	for i = 1:size(combs, 1)

% the binary vector features is used to zero the contribution of features that are being left out
% when recorded as best, it denotes the best configuration so far
		features = zeros(d, 1);
		features(combs(i, :)) = 1;
		s = sum(cFCorr/sqrt(sqrt(sum(sum(ffCorr.*repmat(features, 1, d))))));
		if s > score
			best = features;
			score = s;
		end
	end

	good = find(best == 1)';
