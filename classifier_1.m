function classifier_1(trainpath, testpath, outtrainpath, outtestpath)

% read file and set up variables
	trainFile = fopen(trainpath, 'r');
	C = textscan(trainFile, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(trainFile);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};
	perClass = 16;

	labelSet = unique(sampleLabel);

	good = select_features(sample, perClass, 2)
	sample = sample(:, good);

% normalize using mean and sd
	sampleMu = repmat(mean(sample), size(sample, 1), 1);
	sampleSigma = repmat(sqrt(var(sample)), size(sample, 1), 1);
	sample = (sample - sampleMu)./sampleSigma;

	correct = zeros(perClass, 1);
	bestSuccess = 0;
	classifierLabel = {};
	classifier = [];
	testLabel = {};
	test = [];
	step = round(0.7*perClass);

	for offset = 0:perClass-1

		trainCount = zeros(1, size(labelSet, 1));
		testCount = zeros(1, size(labelSet, 1));
		validate = [];
		train = [];
		validateLabel = {};
		trainLabel = {};

% split samples for cross validation
		i = 1;
		j = 1;

		for k = 1:size(sample, 1)
			if mod(k - offset, perClass) < step
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
			X = train(1+(c-1)*(step):c*(step-1), :);
			mu(c, :) = mean(X);
			k = 1 + (c-1)*size(train, 2);
			sigma(k:k+size(train, 2)-1, :) = cov(X);
		end

%		b = min(min(sigma));
%		if b < 0 sigma = 1 + sigma - b; end

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
				correct(offset+1) = correct(offset+1) + 1;
			end
			validateLabel{i} = class;
		end
		correct(offset+1) = correct(offset+1)/size(validate, 1);
		if correct(offset+1) > bestSuccess
			bestSuccess = correct(offset+1);
			classifierLabel = trainLabel;
			classifier = train;
			testLabel = validateLabel;
			test = validate;
		end

		assert(sum(trainCount) == size(labelSet, 1)*trainCount(1));
		assert(sum(testCount) == size(labelSet, 1)*testCount(1));
	end

	bestSuccess
	correct

	trainFile = fopen(outtrainpath, 'w');
	for i = 1:size(test, 1)
		fprintf(trainFile, '%s %f ', testLabel{i}, test(i,:));
		fprintf(trainFile, '\n');
	end
	fclose(trainFile);

% read and classify testing data
	testFile = fopen(testpath, 'r');
	C = textscan(testFile, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(testFile);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};

	sample = sample(:, good);
	sample = (sample - sampleMu)./sampleSigma;

	testFile = fopen(outtestpath, 'w');
	for i = 1:size(sample, 1)
		c = sampleLabel{i};
		x = sample(i);
		best = Inf;

		for j = 1:size(classifier, 1)
			d = norm(x - classifier(j, :));
			if d < best
				c = classifierLabel{j};
				best = d;
			end
		end
		fprintf(testFile, '%s %f ', c, sample(i, :));
		fprintf(testFile, '\n');
	end
	fclose(testFile);

function [p] = log_mvnpdf(x, m, s)

	c0 = -0.5*size(m, 1)*log(2*pi);
	c1 = -0.5*log(det(s));
	phi = -0.5*(x - m)*pinv(s)*(x - m)';
	p = c0 + c1 + phi;
