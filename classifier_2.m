function classifier_2(trainpath, testpath, outtrainpath, outtestpath)

	% read file and set up variables
	fTrainIn = fopen(trainpath, 'r');
	C = textscan(fTrainIn, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(fTrainIn);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};
	perClass = 16;

	labelSet = unique(sampleLabel);

	for i = 1:size(sample, 2)
		sample(:, i) = (sample(:, i) - mean(sample(:, i)))/sqrt(var(sample(:, i)));
	end

	good = select_features(sample, perClass, 9)
	sample = sample(:, good);
	
	numValidate = ceil(size(sample, 1)/2);
	numTrain = floor(size(sample, 1)/2);

	validate = zeros(numValidate, size(sample, 2));
	train = zeros(numTrain, size(sample, 2));

	validateLabel = {};
	trainLabel = {};

	% split samples for cross validation
	i = 1;
	j = 1;
	for k = 1:size(sample, 1)
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

