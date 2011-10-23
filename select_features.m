% for a given set of samples, discover the features that have the least intra-class variance for each class and return their indices
% t is a threshold
% n is an upper bound for the number of features
function [good] = select_features(sample, step, n)

	assert(n <= size(sample, 2));
	good = ones(size(sample, 2), 1);
	mvar = zeros(size(sample, 2), 1);
	for i = 1:step:size(sample, 1)
		v = var(sample(i:i+step-1, :));
		for j = 1:size(sample, 2)
			if v(j) > mvar(j) mvar(j) = v(j); end
		end
	end	
	[s, i] = sort(mvar);
	good = i(1:n);
