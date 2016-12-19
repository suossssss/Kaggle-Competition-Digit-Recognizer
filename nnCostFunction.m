function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%calculating h(x)
X=[ones(m,1) X];%input layer (add a0(1))

z2=X*Theta1';%seconde layer(Hidden layer)
a2=sigmoid(z2);
a2=[ones(size(a2,1),1) a2];%(add a0(2))

z3=a2*Theta2';
a3=sigmoid(z3);

h=a3;%5000*10

yk=zeros(size(y,1),num_labels);

for i=1:m
    yk(i,y(i))=1;
end

cost=-yk.*log(h)-(1-yk).*log(1-h);

%%regularized cost function
temp1=[zeros(size(Theta1,1),1) Theta1(:,2:end)];
temp2=[zeros(size(Theta2,1),1) Theta2(:,2:end)];
temp1=sum(temp1.^2);
temp2=sum(temp2.^2);
J=sum(cost(:))/m+(lambda/(2*m))*(sum(temp1(:))+sum(temp2(:)));
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

D1=zeros(size(Theta1));
D2=zeros(size(Theta2));


for t=1:m
    %step1 
    a_1=X(t,:);%1*401
    z_2=a_1*Theta1';%1*25
    a_2=sigmoid(z_2);
    a_2=[ones(size(a_2,1),1) a_2];%1*26
    z_3=a_2*Theta2';%1*10
    a_3=sigmoid(z_3);
    %step2
    delta3=zeros(num_labels,1);
    for k=1:num_labels
        delta3(k)=a_3(k)-(y(t)==k);
    end
    
    %step3
    delta2=Theta2'*delta3;
    delta2=delta2(2:end).*sigmoidGradient(z_2)';
    
    %step4
    D2=D2+delta3*a_2;
    D1=D1+delta2*a_1;
end

%step5
Theta1_grad= D1/m+lambda/m*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad= D2/m+lambda/m*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end