function classifier_2(trainpath, testpath, outtrainpath, outtestpath)

% read file and set up variables
	trainFile = fopen(trainpath, 'r');
	C = textscan(trainFile, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(trainFile);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};
	perClass = 16;

	labelSet = unique(sampleLabel);

% select the best features
	good = select_features(sample, perClass, 11);
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
			r = find(ismember(labelSet, sampleLabel{k}) == 1);
			if mod(k - offset, perClass) < round(0.7*perClass)
				trainLabel{end+1} = sampleLabel{k};
				trainCount(r) = trainCount(r) + 1;
				train(i, :) = sample(k, :);
				i = i + 1;
			else
				validateLabel{end+1} = sampleLabel{k};
				testCount(r) = testCount(r) + 1;
				validate(j, :) = sample(k, :);
				j = j + 1;
			end
		end

%classify the test portion of the training data using the nearest neighbor rule
		for i = 1:size(validate, 1)
			x = validate(i, :);
			best = [Inf, 1];
			for j = 1:size(train, 1)
				l = find(ismember(labelSet, trainLabel{j}) == 1);
				y = train(j, :);
				s = norm(x - y);
				if s < best(1)
					best = [s, l];
				end
			end
			class = labelSet{best( 2)};
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
