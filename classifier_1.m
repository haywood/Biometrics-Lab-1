function classifier_1(trainpath, testpath, outtrainpath, outtestpath)

% read file and set up variables
	trainFile = fopen(trainpath, 'r');
	C = textscan(trainFile, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(trainFile);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};
	perClass = 16;

	labelSet = unique(sampleLabel);

% select the best features
	good = select_features(sample, perClass, 9);
	sample = sample(:, good);

% standardize using mean and sd
	sampleMu = repmat(mean(sample), size(sample, 1), 1);
	sampleSigma = repmat(std(sample), size(sample, 1), 1);
	sample = (sample - sampleMu)./sampleSigma;

	numValidations = 10;
	correct = zeros(numValidations, 1);
	bestSuccess = 0;
	classifierLabel = {};
	classifier = [];
	testLabel = {};
	test = [];
	step = round(0.7*perClass);

	for offset = 0:numValidations-1

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

% calculate mean and covariance for each distribution
		sigma = zeros(size(labelSet, 1)*size(train, 2), size(train, 2));
		mu = zeros(size(labelSet, 1), size(train, 2));
		for c = 1:size(labelSet, 1)
			X = train(1+(c-1)*step:c*(step-1), :);
			mu(c, :) = mean(X);
			k = 1 + (c-1)*size(train, 2);
			sigma(k:k+size(train, 2)-1, :) = cov(X);
		end

% make sure each covariance matrix is positive definite
		b = min(min(sigma));
		if b < 0 sigma = 1 + sigma - b; end

% calculate the most likely class for each of the test samples
		best = zeros(size(validate, 1), 2);
		best(:) = -Inf;
		for c = 1:size(labelSet, 1)
			sig = sigma(1+(c-1)*size(train, 2):c*size(train, 2), :);
			for i = 1:size(validate, 1)
				x = validate(i, :);
				m = log_mvnpdf(x, mu(c, :), sig);
				if m >= best(i, 1)
					best(i, :) = [m c];
				end
			end
		end

% assign the classes and update the correct counts
		for i = 1:size(validate, 1)
			class = labelSet{best(i, 2)};
			if strcmp(class, validateLabel{i})
				correct(offset+1) = correct(offset+1) + 1;
			end
			validateLabel{i} = class;
		end

% calculate fraction correct and update the most accurate classifier so far
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

	display 'correct counts and average for training'
	correct % show the fraction correct for each classifier
	mean(correct) % show the average

	trainFile = fopen(outtrainpath, 'w');
	for i = 1:size(test, 1)
		fprintf(trainFile, '%s %f ', testLabel{i}, test(i,:));
		fprintf(trainFile, '\n');
	end
	fclose(trainFile);

% read and classify the real testing data
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

% calculate the natural logarithm of the multivariate normal distribution
function [p] = log_mvnpdf(x, m, s)

	c0 = -0.5*size(m, 1)*log(2*pi);
	c1 = -0.5*log(det(s));
	phi = -0.5*(x - m)*pinv(s)*(x - m)';
	p = c0 + c1 + phi;
