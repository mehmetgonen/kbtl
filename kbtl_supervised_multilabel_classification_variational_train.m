% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbtl_supervised_multilabel_classification_variational_train(K, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    T = length(K);
    D = zeros(T, 1);
    N = zeros(T, 1);
    for t = 1:T
        D(t) = size(K{t}, 1);
        N(t) = size(K{t}, 2);
    end
    L = size(Y{1}, 2);
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

    gamma.alpha = (parameters.alpha_gamma + 0.5) * ones(L, 1);
    gamma.beta = parameters.beta_gamma * ones(L, 1);
    Eta.alpha = (parameters.alpha_eta + 0.5) * ones(R, L);
    Eta.beta = parameters.beta_eta * ones(R, L);
    bW.mu = [zeros(1, L); randn(R, L)];
    bW.sigma = repmat(eye(R + 1, R + 1), [1, 1, L]);

    F = cell(1, T);
    for t = 1:T
        F{t}.mu = (abs(randn(N(t), L)) + parameters.margin) .* sign(Y{t});
        F{t}.sigma = ones(N(t), L);
    end

    KKT = cell(1, T);
    for t = 1:T
        KKT{t} = K{t} * K{t}';
    end

    lower = cell(1, T);
    upper = cell(1, T);
    for t = 1:T
        lower{t} = -1e40 * ones(N(t), L);
        lower{t}(Y{t} > 0) = +parameters.margin;
        upper{t} = +1e40 * ones(N(t), L);
        upper{t}(Y{t} < 0) = -parameters.margin;
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
            H{t}.sigma = (eye(R, R) / sigma_h^2 + bW.mu(2:R + 1, :) * bW.mu(2:R + 1, :)' + sum(bW.sigma(2:R + 1, 2:R + 1, :), 3)) \ eye(R, R);
            H{t}.mu = A{t}.mu' * K{t} / sigma_h^2;
            for o = 1:L
                H{t}.mu = H{t}.mu + bW.mu(2:R + 1, o) * F{t}.mu(:, o)' - repmat(bW.mu(2:R + 1, o) * bW.mu(1, o) + bW.sigma(2:R + 1, 1, o), 1, N(t));
            end
            H{t}.mu = H{t}.sigma * H{t}.mu;
        end

        for o = 1:L
            %%%% update gamma        
            gamma.beta(o) = 1 / (1 / parameters.beta_gamma + 0.5 * (bW.mu(1, o)^2 + bW.sigma(1, 1, o)));
            %%%% update Eta
            Eta.beta(:, o) = 1 ./ (1 / parameters.beta_eta + 0.5 * (bW.mu(2:R + 1, o).^2 + diag(bW.sigma(2:R + 1, 2:R + 1, o))));
            %%%% update b and W
            bW.sigma(:, :, o) = [gamma.alpha(o) * gamma.beta(o), zeros(1, R); zeros(R, 1), diag(Eta.alpha(:, o) .* Eta.beta(:, o))];
            for t = 1:T
                bW.sigma(:, :, o) = bW.sigma(:, :, o) + [N(t), sum(H{t}.mu, 2)';
                                                                   sum(H{t}.mu, 2), H{t}.mu * H{t}.mu' + N(t) * H{t}.sigma];
            end
            bW.sigma(:, :, o) = bW.sigma(:, :, o) \ eye(R + 1, R + 1);
            bW.mu(:, o) = zeros(R + 1, 1);
            for t = 1:T
                bW.mu(:, o) = bW.mu(:, o) + [ones(1, N(t)); H{t}.mu] * F{t}.mu(:, o);
            end
            bW.mu(:, o) = bW.sigma(:, :, o) * bW.mu(:, o);
        end

        for t = 1:T
            %%%% update F
            output = [ones(1, N(t)); H{t}.mu]' * bW.mu;
            alpha_norm = lower{t} - output;
            beta_norm = upper{t} - output;
            normalization = normcdf(beta_norm) - normcdf(alpha_norm);
            normalization(normalization == 0) = 1;
            F{t}.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
            F{t}.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
        end
    end

    state.Lambda = Lambda;
    state.A = A;
    state.gamma = gamma;
    state.Eta = Eta;
    state.bW = bW;
    state.parameters = parameters;
end