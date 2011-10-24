function [good] = select_features_1(sample, step, k)

	assert(k > 0 & k < size(sample, 2));

	d = size(sample, 2);
	features = ones(d, 1);
	best = features;
	score = -Inf;

	ffCorr = corr(sample);
	cfCorr = zeros(1, d);

	for i = 1:size(sample, 1)
		sample(i, d+1) = ceil(i/step);
	end

	for i = 1:d
		cFCorr(i) = corr(sample(:, i), sample(:, d+1));
	end

	for n = 1:k
		combs = nchoosek(1:d, n);
		for i = 1:size(combs, 1)
			features = zeros(d, 1);
			features(combs(i, :)) = 1;
			s = sum(cFCorr/sqrt(sqrt(sum(sum(ffCorr.*repmat(features, 1, d))))));
			if s > score
				best = features;
				score = s;
			end
		end
	end

	good = find(best == 1)';
