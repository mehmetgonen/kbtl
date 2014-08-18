% Mehmet Gonen (mehmet.gonen@gmail.com)

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

    gamma.shape = (parameters.alpha_gamma + 0.5);
    gamma.scale = parameters.beta_gamma;
    eta.shape = (parameters.alpha_eta + 0.5) * ones(R, 1);
    eta.scale = parameters.beta_eta * ones(R, 1);
    bw.mean = [0; randn(R, 1)];
    bw.covariance = eye(R + 1, R + 1);

    f = cell(1, T);
    for t = 1:T
        f{t}.mean = (abs(randn(N(t), 1)) + parameters.margin) .* sign(y{t});
        f{t}.covariance = ones(N(t), 1);
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
                Lambda{t}.scale(:, s) = 1 ./ (1 / parameters.beta_lambda + 0.5 * (A{t}.mean(:, s).^2 + diag(A{t}.covariance(:, :, s))));
            end
            %%%% update A
            for s = 1:R
                A{t}.covariance(:, :, s) = (diag(Lambda{t}.shape(:, s) .* Lambda{t}.scale(:, s)) + KKT{t} / sigmah^2) \ eye(D(t), D(t));
                A{t}.mean(:, s) = A{t}.covariance(:, :, s) * (K{t} * H{t}.mean(s, :)' / sigmah^2);
            end
            %%%% update H
            H{t}.covariance = (eye(R, R) / sigmah^2 + bw.mean(2:R + 1) * bw.mean(2:R + 1)' + bw.covariance(2:R + 1, 2:R + 1)) \ eye(R, R);
            H{t}.mean = H{t}.covariance * (A{t}.mean' * K{t} / sigmah^2 + bw.mean(2:R + 1) * f{t}.mean' - repmat(bw.mean(2:R + 1) * bw.mean(1) + bw.covariance(2:R + 1, 1), 1, N(t)));
        end

        %%%% update gamma
        gamma.scale = 1 / (1 / parameters.beta_gamma + 0.5 * (bw.mean(1)^2 + bw.covariance(1, 1)));
        %%%% update eta
        eta.scale = 1 ./ (1 / parameters.beta_eta + 0.5 * (bw.mean(2:R + 1).^2 + diag(bw.covariance(2:R + 1, 2:R + 1))));
        %%%% update b and w
        bw.covariance = [gamma.shape * gamma.scale, zeros(1, R); zeros(R, 1), diag(eta.shape .* eta.scale)];
        for t = 1:T
            bw.covariance = bw.covariance + [N(t), sum(H{t}.mean, 2)';
                                             sum(H{t}.mean, 2), H{t}.mean * H{t}.mean' + N(t) * H{t}.covariance];
        end
        bw.covariance = bw.covariance \ eye(R + 1, R + 1);
        bw.mean = zeros(R + 1, 1);
        for t = 1:T
            bw.mean = bw.mean + [ones(1, N(t)); H{t}.mean] * f{t}.mean;
        end
        bw.mean = bw.covariance * bw.mean;

        for t = 1:T
            %%%% update f
            output = [ones(1, N(t)); H{t}.mean]' * bw.mean;
            alpha_norm = lower{t} - output;
            beta_norm = upper{t} - output;
            normalization = normcdf(beta_norm) - normcdf(alpha_norm);
            normalization(normalization == 0) = 1;
            f{t}.mean = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
            f{t}.covariance = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
        end
    end

    state.Lambda = Lambda;
    state.A = A;
    state.gamma = gamma;
    state.eta = eta;
    state.bw = bw;
    state.parameters = parameters;
end