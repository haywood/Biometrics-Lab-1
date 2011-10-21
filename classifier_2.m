function classifier_2(trainpath, testpath, outtrainpath, outtestpath)

% read file and set up variables
	fTrainIn = fopen(trainpath, 'r');
	C = textscan(fTrainIn, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(fTrainIn);

	sample = cell2mat(C(2:end));
	sampleLabel = C{1};
	perClass = 16;

	labelSet = unique(sampleLabel);

% normalize using mean and sd
	o = repmat(mean(sample), size(sample, 1), 1);
	d = repmat(sqrt(var(sample)), size(sample, 1), 1);
	sample = (sample - o)./d;

	good = select_features(sample, perClass, 11);
	sample = sample(:, good);
	
	train = zeros(size(sample, 1) - size(labelSet, 1), size(sample, 2));
	validate = zeros(size(labelSet, 1), size(sample, 2));

	correct = 0;

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

		k = 1;
		for i = 1:size(validate, 1)
			best = [Inf, 1];
			for j = 1:size(train, 1)
				l = find(ismember(labelSet, trainLabel{j}) == 1);
				m = norm(validate(i, :) - train(j, :));
				if m < best(1)
					best = [m, l];
				end
			end
			class = labelSet{best( 2)};
			if strcmp(class, validateLabel{i})
				correct = correct + 1;
			end
			validateLabel{i} = class;
		end
	end

	correct/size(sample, 1)

	fTrainOut = fopen(outtrainpath, 'w');
	for ix = 1:size(validate, 1)
		fprintf(fTrainOut, '%s', validateLabel{ix});
		fprintf(fTrainOut, ' %f', validate(ix,:));
		fprintf(fTrainOut, '\n');
	end
	fclose(fTrainOut);
