% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbtl_supervised_multilabel_classification_variational_test(K, state)
    T = length(K);
    N = zeros(T, 1);
    for t = 1:T
        N(t) = size(K{t}, 2);
    end
    L = size(state.bW.sigma, 3);

    prediction.H = cell(1, T);
    for t = 1:T
        if N(t) > 0
            prediction.H{t}.mu = state.A{t}.mu' * K{t};
        end
    end

    prediction.F = cell(1, T);
    for t = 1:T
        if N(t) > 0
            prediction.F{t}.mu = [ones(1, N(t)); prediction.H{t}.mu]' * state.bW.mu;
            prediction.F{t}.sigma = zeros(N(t), L);
            for o = 1:L
                prediction.F{t}.sigma(:, o) = 1 + diag([ones(1, N(t)); prediction.H{t}.mu]' * state.bW.sigma(:, :, o) * [ones(1, N(t)); prediction.H{t}.mu]);
            end
        end
    end
    
    prediction.P = cell(1, T);
    for t = 1:T
        if N(t) > 0
            pos = 1 - normcdf((+state.parameters.margin - prediction.F{t}.mu) ./ prediction.F{t}.sigma);
            neg = normcdf((-state.parameters.margin - prediction.F{t}.mu) ./ prediction.F{t}.sigma);
            prediction.P{t} = pos ./ (pos + neg);
        end
    end
end