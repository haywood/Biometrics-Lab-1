function [good] = select_features_1(sample, step, n)

	assert(n <= size(sample, 2));
	fVar = zeros(2, size(sample, 2));

	for i = 1:size(sample, 2)
		for j = 1:step:size(sample, 1)
			X = sample(j:j+step-1, :);
			intraClass = var(X(:,i));
			if intraClass > fVar(2, i)
				fVar(2, i) = intraClass;
			end
			for k = 1:step:size(sample, 1)
				if k < j
					Y = sample(k:k+step-1, :);
					Axy = [X(:,i); Y(:,i)];
					interClass = var(Axy);
					if interClass > fVar(1, i)
						fVar(1, i) = interClass;
					end
				end
			end
		end
	end

	[s, i] = sort(diff(fVar), 'descend')
	good = i(1:n);
