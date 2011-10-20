function classifier_2(trainpath, testpath, outtrainpath, outtestpath)

	% read file and set up variables
	fTrainIn = fopen(trainpath, 'r');
	C = textscan(fTrainIn, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(fTrainIn);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};
	perClass = 16;

	numLabel = size(sampleLabel, 1);
	numSample = size(sample, 1);
	sampleOrd = 1:numSample;

	labelSet = unique(sampleLabel);
	numLabel = size(labelSet, 1);

	for i = 1:numSample
		j = find(ismember(labelSet, sampleLabel{i}) == 1);
		sampleOrd(i) = j;
	end

	for i = 1:numSample
		sample(i, :) = sample(i, :)/norm(sample(i, :));
	end

%	for i = 1:size(sample, 2)
%		var(sample(:, i)/sqrt(var(sample(:, i))))
%	end

	good = select_features(sample, perClass, 0, 0.001);
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

	correct = 0;
	k = 1;
	for i = 1:numValidate
		near = [];
		for j = 1:numTrain
			l = find(ismember(labelSet, trainLabel{j}) == 1);
			m = norm(validate(i, :) - train(j, :));
			if size(near, 1) < k
				near = [near; m l];
			elseif m < near(k, 1)
				near(k, :) = [m l];
			end
			t = size(near, 1) - 1;
			while t > 0 && near(t, 1) > near(t+1, 1)
				tmp = near(t, :);
				near(t, :) = near(t+1, :);
				near(t+1, :) = tmp;
			end
		end
		class = labelSet{mode(near(:, 2))};
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

