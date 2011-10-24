function classifier_1(trainpath, testpath, outtrainpath, outtestpath)

% read file and set up variables
	fTrainIn = fopen(trainpath, 'r');
	C = textscan(fTrainIn, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(fTrainIn);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};
	perClass = 16;

	labelSet = unique(sampleLabel);

	mu = repmat(mean(sample), size(sample, 1), 1);
	sigma = repmat(sqrt(var(sample)), size(sample, 1), 1);
	sample = (sample - mu)./sigma;

	good = select_features(sample, perClass, 11)
	sample = sample(:, good);

% whitening transform
	[V, D] = eig(cov(sample));
	Aw = V*D^(-0.5);
	mu = repmat(mean(sample), size(sample, 1), 1);
	sample = (Aw*sample')';

	train = zeros(size(sample, 1) - size(labelSet, 1), size(sample, 2));
	validate = zeros(size(labelSet, 1), size(sample, 2));
	correct = zeros(perClass-1, 1);

	for offset = 1:perClass-1

		validateLabel = {};
		trainLabel = {};

% split samples for cross validation
		i = 1;
		j = 1;
		for k = 1:size(sample, 1)
			if mod(k - offset, perClass)
				trainLabel{end+1} = sampleLabel{k};
				train(i, :) = sample(k, :);
				i = i + 1;
			else
				validateLabel{end+1} = sampleLabel{k};
				validate(j, :) = sample(k, :);
				j = j + 1;
			end
		end

		sigma = zeros(size(labelSet, 1)*size(train, 2), size(train, 2));
		mu = zeros(size(labelSet, 1), size(train, 2));
		for c = 1:size(labelSet, 1)
			X = train(1+(c-1)*(perClass-1):c*(perClass-2), :);
			mu(c, :) = mean(X);
			k = 1 + (c-1)*size(train, 2);
			sigma(k:k+size(train, 2)-1, :) = cov(X);
		end

		b = min(min(sigma));
		if b < 0 sigma = 1 + sigma - b; end

		best = zeros(size(validate, 1), 2);
		best(:) = -Inf;
		for c = 1:size(labelSet, 1)
			sig = sigma(1+(c-1)*size(train, 2):c*size(train, 2), :);
			assert(all(all(sig > 0))) % make sure the sample covariance matrix is positive definite
			for i = 1:size(validate, 1)
				x = validate(i, :);
				m = log_mvnpdf(x, mu(c, :), sig);
				if m >= best(i, 1)
					best(i, :) = [m c];
				end
			end
		end

		for i = 1:size(validate, 1)
			class = labelSet{best(i, 2)};
			if strcmp(class, validateLabel{i})
				correct(offset) = correct(offset) + 1;
			end
			validateLabel{i} = class;
		end
	end

	correct/size(labelSet, 1)
	[s, i] = sort(correct, 'descend');
	i(1)

	fTrainOut = fopen(outtrainpath, 'w');
	for ix = 1:size(validate, 1)
		fprintf(fTrainOut, '%s', validateLabel{ix});
		fprintf(fTrainOut, ' %f', validate(ix,:));
		fprintf(fTrainOut, '\n');
	end
	fclose(fTrainOut);

function [p] = log_mvnpdf(x, m, s)

	c0 = -0.5*size(m, 1)*log(2*pi);
	c1 = -0.5*log(det(s));
	phi = -0.5*(x - m)*pinv(s)*(x - m)';
	p = c0 + c1 + phi;
