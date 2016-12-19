
tic
% % %load data
train=csvread('train.csv',1,0);
X_test=csvread('test.csv',1,0);


Y_train=train(:,1);
X_train=train(:,2:end);

for i=1:size(Y_train)
    if Y_train(i)==0
        Y_train(i)=10;
    end
end

% normalize dataset
[X_train_norm,mu,sigma]=featureNormalize(X_train);

X_train_norm(isnan(X_train_norm))=0;

% Randomly select 100 data points to display
m = size(X_train, 1);

sel = randperm(m);
sel = sel(1:100);

displayData(X_train_norm(sel, :));

fprintf('Visualizing the data.\n');


input_layer_size=784;
hidden_layer_size=80;
num_labels=10;


fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 500);

%  You should also try different values of lambda
lambda = 6;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train_norm, Y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred_train= predict(Theta1, Theta2, X_train_norm);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == Y_train)) * 100);

fprintf('\nPredicting the final result...\n');

[X_test_norm,mu,sigma]=featureNormalize(X_test);

X_test_norm(isnan(X_test_norm))=0;

pred = predict(Theta1, Theta2, X_test_norm);

for i=1:size(pred)
    if pred(i)==10
        pred(i)=0;
    end
end
fprintf('\nDone!\n');

toc
