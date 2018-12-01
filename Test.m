
fileID = fopen('centers.txt','r'); 
sizeX = [2 Inf];
formatSpec = '%f %f';

X = fscanf(fileID,formatSpec,sizeX);
X = X';

%%====================================================
%% STEP 1: Choose initial values for the parameters.

% Set 'm' to the number of data points.
m = size(X, 1);

k = 3;  % The number of clusters.
n = 2;  % The vector lengths.

% Randomly select k data points to serve as the initial means.
indeces = randperm(m);
mu = X(indeces(1:k), :);

sigma = [];

% Use the overall covariance of the dataset as the initial variance for each cluster.
for (j = 1 : k)
    sigma{j} = cov(X);
end

% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1 / k);

%%===================================================
%% STEP 2: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
W = zeros(m, k);

% Loop until convergence.
for (iter = 1:1000)
    
    fprintf('  EM Iteration %d\n', iter);

    %%===============================================
    %% STEP 2a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(m, k);
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(X, mu(j, :), sigma{j});
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]   
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, phi);
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
    
    %%===============================================
    %% STEP 2b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.

    % Store the previous means.
    prevMu = mu;    
    
    % For each of the clusters...
    for (j = 1 : k)
    
        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(W(:, j), 1);
        
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        mu(j, :) = weightedAverage(W(:, j), X);

        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example. 
        
        sigma_k = zeros(n, n);
        
        % Subtract the cluster mean from all data points.
        Xm = bsxfun(@minus, X, mu(j, :));
        
        % Calculate the contribution of each training example to the covariance matrix.
        for (i = 1 : m)
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
        end
        
        % Divide by the sum of weights.
        sigma{j} = sigma_k ./ sum(W(:, j));
    end
    
    % Check for convergence.
    if (mu == prevMu)
        break
    end
            
% End of Expectation Maximization    
end

%%=====================================================
%% STEP 3: Plot the data points and their estimated pdfs.

% Display a scatter plot of the two distributions.
figure(2);
hold off;
plot(X(:, 1), X(:, 2), 'bo');
hold on;

set(gcf,'color','white') % White background for the figure.

% 
 % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
 % the input values over the grid.
 gridSize = 1000;
 u = linspace(0, 3000, gridSize);
 [A B] = meshgrid(u, u);
 gridX = [A(:), B(:)];
 
 % Calculate the Gaussian response for every value in the grid.
 z1 = gaussianND(gridX, mu(1, :), sigma{1});
 z2 = gaussianND(gridX, mu(2, :), sigma{2});
 z3 = gaussianND(gridX, mu(3, :), sigma{3}); 
% z4 = gaussianND(gridX, mu(4, :), sigma{4});
% z5 = gaussianND(gridX, mu(4, :), sigma{5});
 
 % Reshape the responses back into a 2D grid to be plotted with contour.
 Z1 = reshape(z1, gridSize, gridSize);
 Z2 = reshape(z2, gridSize, gridSize);
 Z3 = reshape(z3, gridSize, gridSize);
% Z4 = reshape(z4, gridSize, gridSize);
% Z5 = reshape(z5, gridSize, gridSize);
 
 % Plot the contour lines to show the pdf over the data.
 [C, h] = contour(u, u, Z1, 'LineWidth', 3);
 [C, h] = contour(u, u, Z2, 'LineWidth', 3);
 [C, h] = contour(u, u, Z3, 'LineWidth', 3);
% [C, h] = contour(u, u, Z4);
% [C, h] = contour(u, u, Z5);
 axis ij
 axis([0 3000 0 1400])
 
 title('Original Data and Estimated PDFs');