function state = kbtl_supervised_classification_variational_train(K, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    T = length(K);
    D = zeros(T, 1);
    N = zeros(T, 1);
    for t = 1:T
        D(t) = size(K{t}, 1);
        N(t) = size(K{t}, 2);
    end
    R = parameters.R;
    sigma_h = parameters.sigma_h;

    Lambda = cell(1, T);
    A = cell(1, T);
    H = cell(1, T);
    for t = 1:T
        Lambda{t}.alpha = (parameters.alpha_lambda + 0.5) * ones(D(t), R);
        Lambda{t}.beta = parameters.beta_lambda * ones(D(t), R);
        A{t}.mu = randn(D(t), R);
        A{t}.sigma = repmat(eye(D(t), D(t)), [1, 1, R]);
        H{t}.mu = randn(R, N(t));
        H{t}.sigma = eye(R, R);
    end

    gamma.alpha = (parameters.alpha_gamma + 0.5);
    gamma.beta = parameters.beta_gamma;
    eta.alpha = (parameters.alpha_eta + 0.5) * ones(R, 1);
    eta.beta = parameters.beta_eta * ones(R, 1);
    bw.mu = [0; randn(R, 1)];
    bw.sigma = eye(R + 1, R + 1);

    f = cell(1, T);
    for t = 1:T
        f{t}.mu = (abs(randn(N(t), 1)) + parameters.margin) .* sign(y{t});
        f{t}.sigma = ones(N(t), 1);
    end

    KKT = cell(1, T);
    for t = 1:T
        KKT{t} = K{t} * K{t}';
    end

    lower = cell(1, T);
    upper = cell(1, T);
    for t = 1:T
        lower{t} = -1e40 * ones(N(t), 1);
        lower{t}(y{t} > 0) = +parameters.margin;
        upper{t} = +1e40 * ones(N(t), 1);
        upper{t}(y{t} < 0) = -parameters.margin;
    end

    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        for t = 1:T
            %%%% update Lambda
            for s = 1:R
                Lambda{t}.beta(:, s) = 1 ./ (1 / parameters.beta_lambda + 0.5 * (A{t}.mu(:, s).^2 + diag(A{t}.sigma(:, :, s))));
            end
            %%%% update A
            for s = 1:R
                A{t}.sigma(:, :, s) = (diag(Lambda{t}.alpha(:, s) .* Lambda{t}.beta(:, s)) + KKT{t} / sigma_h^2) \ eye(D(t), D(t));
                A{t}.mu(:, s) = A{t}.sigma(:, :, s) * (K{t} * H{t}.mu(s, :)' / sigma_h^2);
            end
            %%%% update H
            H{t}.sigma = (eye(R, R) / sigma_h^2 + bw.mu(2:R + 1) * bw.mu(2:R + 1)' + bw.sigma(2:R + 1, 2:R + 1)) \ eye(R, R);
            H{t}.mu = H{t}.sigma * (A{t}.mu' * K{t} / sigma_h^2 + bw.mu(2:R + 1) * f{t}.mu' - repmat(bw.mu(2:R + 1) * bw.mu(1) + bw.sigma(2:R + 1, 1), 1, N(t)));
        end

        %%%% update gamma
        gamma.beta = 1 / (1 / parameters.beta_gamma + 0.5 * (bw.mu(1)^2 + bw.sigma(1, 1)));
        %%%% update eta
        eta.beta = 1 ./ (1 / parameters.beta_eta + 0.5 * (bw.mu(2:R + 1).^2 + diag(bw.sigma(2:R + 1, 2:R + 1))));
        %%%% update b and w
        bw.sigma = [gamma.alpha * gamma.beta, zeros(1, R); zeros(R, 1), diag(eta.alpha .* eta.beta)];
        for t = 1:T
            bw.sigma = bw.sigma + [N(t), sum(H{t}.mu, 2)';
                                             sum(H{t}.mu, 2), H{t}.mu * H{t}.mu' + N(t) * H{t}.sigma];
        end
        bw.sigma = bw.sigma \ eye(R + 1, R + 1);
        bw.mu = zeros(R + 1, 1);
        for t = 1:T
            bw.mu = bw.mu + [ones(1, N(t)); H{t}.mu] * f{t}.mu;
        end
        bw.mu = bw.sigma * bw.mu;

        for t = 1:T
            %%%% update f
            output = [ones(1, N(t)); H{t}.mu]' * bw.mu;
            alpha_norm = lower{t} - output;
            beta_norm = upper{t} - output;
            normalization = normcdf(beta_norm) - normcdf(alpha_norm);
            normalization(normalization == 0) = 1;
            f{t}.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
            f{t}.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
        end
    end

    state.Lambda = Lambda;
    state.A = A;
    state.gamma = gamma;
    state.eta = eta;
    state.bw = bw;
    state.parameters = parameters;
end
