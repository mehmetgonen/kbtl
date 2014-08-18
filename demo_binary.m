%initalize the parameters of the algorithm
parameters = struct();

%set the hyperparameters of gamma prior used for projection matrices
parameters.alpha_lambda = 1;
parameters.beta_lambda = 1;

%set the hyperparameters of gamma prior used for bias
parameters.alpha_gamma = 1;
parameters.beta_gamma = 1;

%set the hyperparameters of gamma prior used for weights
parameters.alpha_eta = 1;
parameters.beta_eta = 1;

%%% IMPORTANT %%%
%For gamma priors, you can experiment with three different (alpha, beta) values
%(1, 1) => default priors
%(1e-10, 1e+10) => good for obtaining sparsity
%(1e-10, 1e-10) => good for small sample size problems

%set the number of iterations
parameters.iteration = 200;

%set the margin parameter
parameters.margin = 0;

%set the subspace dimensionality
parameters.R = 2;

%set the seed for random number generator used to initalize random variables
parameters.seed = 1606;

%set the standard deviation of hidden representations
parameters.sigmah = 0.1;

%set the number of tasks
T = ??;

%initialize the kernel and class labels of each task for training
Ktrain = cell(1, T);
ytrain = cell(1, T);
for t = 1:T
    Ktrain{t} = ??; %should be an Ntra x Ntra matrix containing similarity values between training samples of task t
    ytrain{t} = ??; %should be an Ntra x 1 matrix containing class labels of task t (contains only -1s and +1s)
end

%perform training
state = kbtl_supervised_classification_variational_train(Ktrain, ytrain, parameters);

%initialize the kernel of each task for testing
Ktest = cell(1, T);
for t = 1:T
    Ktest{t} = ??; %should be an Ntra x Ntest matrix containing similarity values between training and test samples of task t
end

%perform prediction
prediction = kbtl_supervised_classification_variational_test(Ktest, state);

%display the predicted probabilities for each task
for t = 1:T
    display(prediction.p{t});
end
