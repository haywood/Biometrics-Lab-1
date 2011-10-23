function classifier_1(trainpath, testpath, outtrainpath, outtestpath)

% read file and set up variables
	fTrainIn = fopen(trainpath, 'r');
	C = textscan(fTrainIn, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(fTrainIn);

	sample = cell2mat(C([3 4 5 9 10 12 14]));
	sampleLabel = C{1};
	perClass = 16;

	labelSet = unique(sampleLabel);

% whitening transform
	[V, D] = eig(cov(sample));
	Aw = V*D^(-0.5);
	mu = repmat(mean(sample), size(sample, 1), 1);
	sample = (Aw*(sample - mu)')';

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

		best = zeros(size(validate, 1), 2);
		for c = 1:size(labelSet, 1)
			X = train(1+(c-1)*(perClass-1):c*(perClass-2), :);
			mu = mean(X);
			sigma = cov(X);
			for i = 1:size(validate, 1)
				m = mvnpdf(validate(i, :), mu, sigma);
				if m >= best(i, 1)
					best(i, :) = [m c];
				end
			end
		end

		for i = 1:size(validate, 1)
			class = labelSet{best(i, 2)};
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
