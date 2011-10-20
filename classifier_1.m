function classifier_1(trainpath, testpath, outtrainpath, outtestpath)

	% read file and set up variables
	fTrainIn = fopen(trainpath, 'r');
	C = textscan(fTrainIn, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(fTrainIn);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};
	perClass = 16;

	numLabel = size(sampleLabel, 1);
	numSample = size(sample, 1);

	labelSet = unique(sampleLabel);
	numLabel = size(labelSet, 1);

	for i = 1:size(sample, 2)
		sample(:, i) = (sample(:, i) - mean(sample(:, i)))/sqrt(var(sample(:, i)));
	end

	good = select_features(sample, perClass, 3);
	sample = sample(:, good);

	numValidate = ceil(numSample/2);
	numTrain = floor(numSample/2);
	numFeat = size(sample, 2);

	validate = zeros(numValidate, numFeat);
	train = zeros(numLabel, numFeat);

	validateLabel = {};
	trainLabel = {};

	% split samples for cross validation
	i = 1;
	j = 1;
	for k = 1:numSample
		if mod(k, perClass) < floor(perClass/2)
			trainLabel{end+1} = sampleLabel{k};
			train(i, :) = sample(k, :);
			i = i + 1;
		else
			validateLabel{end+1} = sampleLabel{k};
			validate(j, :) = sample(k, :);
			j = j + 1;
		end
	end

	% calculate mean and cov of the training data
	mu = zeros(numLabel, numFeat);
	sigma = zeros(numFeat*numLabel, numFeat);
	for i = 1:numTrain
		j = find(ismember(labelSet, trainLabel{i}));
		mu(j, :) = mu(j, :) + train(i, :);
	end
	mu = mu/numTrain;

	work = zeros(numTrain/numLabel, numFeat);
	for i = 1:numLabel
		k = 1;
		for j = 1:numTrain
			if strcmp(trainLabel{j}, labelSet{i})
				work(k, :) = train(j, :);
				k = k + 1;
			end
		end
		sigma(1+(i-1)*numFeat:i*numFeat, :) = cov(work);
	end

	correct = 0;
	for i = 1:numValidate
		best = -Inf;
		class = '';
		for j = 1:numLabel
			m = P(validate(i, :), mu(j, :), sigma(1+(j-1)*numFeat:j*numFeat, :));
			if m > best
				class = labelSet{j};
				best = m;
			end
		end
		if strcmp(class, validateLabel{i})
			correct = correct + 1;
		end
	end

	correct/numValidate

	fTrainOut = fopen(outtrainpath, 'w');
	for ix = 1:numValidate
		fprintf(fTrainOut, '%s', validateLabel{ix});
		fprintf(fTrainOut, ' %f', validate(ix,:));
		fprintf(fTrainOut, '\n');
	end
	fclose(fTrainOut);

function [p] = P(x, mu, sigma)
	d = size(mu, 1); % dimensionality
	p = exp(-(x - mu)*inv(sigma)*(x - mu)'/2)/((2*pi*det(sigma))^(d/2));
