function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
rand = zeros(2,1);

for iter = 1:num_iters
    for j = 1:2
    S=0;
    for i = 1:m
        h = theta(1,1)*X(i,1)+theta(2,1)*X(i,2) ;
        S = S + (h-y(i))*X(i,j)/m;
    end
    rand(j,1) = theta(j,1) - alpha*S;
    end
    theta = rand ;
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('%f\n',J_history(iter));

end

end
