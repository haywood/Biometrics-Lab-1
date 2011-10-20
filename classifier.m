% Anything on a line after a '%' is a comment.

% The name of the file should be the name of the function, with ".m"
% appended.  So in this case, the file is classifier_1.m.  This is important
% because Matlab actually looks for a file name, not a function name, when
% you call a function.
% (If you have more than one function in a file, only the first can be called
% from outside that file.)
function classifier_1(trainpath, testpath, outtrainpath, outtestpath)
% If 'trainfile.txt' and 'testfile.txt' are in the current directory, you can
% call this function by typing the following at the Matlab prompt:
% >>  classifier_1('trainfile.txt', 'testfile.txt', 'trainout.txt', 'testout.txt');

% To open a file, use fopen.  It returns a file identifier.  The second
% argument is 'r' for reading or 'w' for writing.
% By the way, Matlab has comprehensive built-in help.  Type
% >> help fopen
% for help with fopen, or
% >> doc fopen
% for the same text in a nice documentation browser.
fTrainIn = fopen(trainpath, 'r');

% We can read the training file with textscan.  The second argument here
% describes the format of each line of the file: a string followed by
% fourteen floating point numbers.
C = textscan(fTrainIn, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f');

% We've read everything we need from the file, so close it.
fclose(fTrainIn);

% C is a "cell array", which is an array that can hold data of any type.  It
% has 15 elements, one for each column in the text file, arranged in a single
% row, so its size is 1 x 15.
fprintf('Size of C: %d x %d\n', size(C));

% We can get the training labels from the first column.  Index into a cell
% array using {}.  Since it's a single row, we can use a single index (C{1,1}
% would also work).
trainLabels = C{1};

% trainLabels is itself a cell array, holding the 96 training labels as
% strings.
fprintf('Size of trainLabels: %d x %d\n', size(trainLabels));
fprintf('First training label: %s\n', trainLabels{1});

% Cell arrays are not great for doing math, so we'll extract the feature
% values from C into a matrix.  Here we use range indexing in the expression
% C(2:end).  Cell array range indexing uses () instead of {}.
trainFeats = cell2mat(C(2:end));

% trainFeats is now a 96 x 14 matrix.  Each row is a sample, and each column
% is a feature.
trainSampleCount = size(trainFeats, 1);
featCount = size(trainFeats, 2);
fprintf('Size of trainFeats: %d x %d\n', trainSampleCount, featCount);

% Index into a matrix using ().  The ij-th entry (i-th row, j-th column) of
% matrix A is A(i,j).
fprintf('The first sample is %s.  Its third feature value is %f.\n', ...
        trainLabels{1}, trainFeats(1,3));

% TODO: Here's where you do your work.  Based on the training labels and feature
% values, create a classifier.  Then read in the test file, ignoring the
% labels (which are all 'test').  Use the classifier and the feature values
% to populate two cell arrays of labels, one using the training feature
% values and one using.
% My classifier is terrible.  It thinks everything as 'raffia'.
trainSampleCount = size(trainLabels, 1); % 96 in this case
trainOutLabels = cell(trainSampleCount, 1);
for ix = 1:trainSampleCount
    trainOutLabels{ix} = 'raffia';
end

% Write the file with classifications of the training data.
fTrainOut = fopen(outtrainpath, 'w');
for ix1 = 1:trainSampleCount
    fprintf(fTrainOut, '%s', trainOutLabels{ix});
    fprintf(fTrainOut, ' %f', trainFeats(ix,:));
    fprintf(fTrainOut, '\n');
end
fclose(fTrainOut);

% TODO: Output the classifications of the test data in the same way.

% That's it!  For further introductory notes, consult the "Getting Started"
% section of the Matlab help.  There are also some good video and text
% tutorials at
% http://www.mathworks.com/academia/student_center/tutorials/launchpad.html.

% Have fun!
