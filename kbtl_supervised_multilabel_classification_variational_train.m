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
    sigmah = parameters.sigmah;

    Lambda = cell(1, T);
    A = cell(1, T);
    H = cell(1, T);
    for t = 1:T
        Lambda{t}.shape = (parameters.alpha_lambda + 0.5) * ones(D(t), R);
        Lambda{t}.scale = parameters.beta_lambda * ones(D(t), R);
        A{t}.mean = randn(D(t), R);
        A{t}.covariance = repmat(eye(D(t), D(t)), [1, 1, R]);
        H{t}.mean = randn(R, N(t));
        H{t}.covariance = eye(R, R);
    end

    gamma.shape = (parameters.alpha_gamma + 0.5) * ones(L, 1);
    gamma.scale = parameters.beta_gamma * ones(L, 1);
    Eta.shape = (parameters.alpha_eta + 0.5) * ones(R, L);
    Eta.scale = parameters.beta_eta * ones(R, L);
    bW.mean = [zeros(1, L); randn(R, L)];
    bW.covariance = repmat(eye(R + 1, R + 1), [1, 1, L]);

    F = cell(1, T);
    for t = 1:T
        F{t}.mean = (abs(randn(N(t), L)) + parameters.margin) .* sign(Y{t});
        F{t}.covariance = ones(N(t), L);
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
                Lambda{t}.scale(:, s) = 1 ./ (1 / parameters.beta_lambda + 0.5 * (A{t}.mean(:, s).^2 + diag(A{t}.covariance(:, :, s))));
            end
            %%%% update A
            for s = 1:R
                A{t}.covariance(:, :, s) = (diag(Lambda{t}.shape(:, s) .* Lambda{t}.scale(:, s)) + KKT{t} / sigmah^2) \ eye(D(t), D(t));
                A{t}.mean(:, s) = A{t}.covariance(:, :, s) * (K{t} * H{t}.mean(s, :)' / sigmah^2);
            end
            %%%% update H
            H{t}.covariance = (eye(R, R) / sigmah^2 + bW.mean(2:R + 1, :) * bW.mean(2:R + 1, :)' + sum(bW.covariance(2:R + 1, 2:R + 1, :), 3)) \ eye(R, R);
            H{t}.mean = A{t}.mean' * K{t} / sigmah^2;
            for o = 1:L
                H{t}.mean = H{t}.mean + bW.mean(2:R + 1, o) * F{t}.mean(:, o)' - repmat(bW.mean(2:R + 1, o) * bW.mean(1, o) + bW.covariance(2:R + 1, 1, o), 1, N(t));
            end
            H{t}.mean = H{t}.covariance * H{t}.mean;
        end

        for o = 1:L
            %%%% update gamma        
            gamma.scale(o) = 1 / (1 / parameters.beta_gamma + 0.5 * (bW.mean(1, o)^2 + bW.covariance(1, 1, o)));
            %%%% update Eta
            Eta.scale(:, o) = 1 ./ (1 / parameters.beta_eta + 0.5 * (bW.mean(2:R + 1, o).^2 + diag(bW.covariance(2:R + 1, 2:R + 1, o))));
            %%%% update b and W
            bW.covariance(:, :, o) = [gamma.shape(o) * gamma.scale(o), zeros(1, R); zeros(R, 1), diag(Eta.shape(:, o) .* Eta.scale(:, o))];
            for t = 1:T
                bW.covariance(:, :, o) = bW.covariance(:, :, o) + [N(t), sum(H{t}.mean, 2)';
                                                                   sum(H{t}.mean, 2), H{t}.mean * H{t}.mean' + N(t) * H{t}.covariance];
            end
            bW.covariance(:, :, o) = bW.covariance(:, :, o) \ eye(R + 1, R + 1);
            bW.mean(:, o) = zeros(R + 1, 1);
            for t = 1:T
                bW.mean(:, o) = bW.mean(:, o) + [ones(1, N(t)); H{t}.mean] * F{t}.mean(:, o);
            end
            bW.mean(:, o) = bW.covariance(:, :, o) * bW.mean(:, o);
        end

        for t = 1:T
            %%%% update F
            output = [ones(1, N(t)); H{t}.mean]' * bW.mean;
            alpha_norm = lower{t} - output;
            beta_norm = upper{t} - output;
            normalization = normcdf(beta_norm) - normcdf(alpha_norm);
            normalization(normalization == 0) = 1;
            F{t}.mean = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
            F{t}.covariance = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
        end
    end

    state.Lambda = Lambda;
    state.A = A;
    state.gamma = gamma;
    state.Eta = Eta;
    state.bW = bW;
    state.parameters = parameters;
end