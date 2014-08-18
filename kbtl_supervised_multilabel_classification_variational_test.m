% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbtl_supervised_multilabel_classification_variational_test(K, state)
    T = length(K);
    N = zeros(T, 1);
    for t = 1:T
        N(t) = size(K{t}, 2);
    end
    L = size(state.bW.covariance, 3);

    prediction.H = cell(1, T);
    for t = 1:T
        if N(t) > 0
            prediction.H{t}.mean = state.A{t}.mean' * K{t};
        end
    end

    prediction.F = cell(1, T);
    for t = 1:T
        if N(t) > 0
            prediction.F{t}.mean = [ones(1, N(t)); prediction.H{t}.mean]' * state.bW.mean;
            prediction.F{t}.covariance = zeros(N(t), L);
            for o = 1:L
                prediction.F{t}.covariance(:, o) = 1 + diag([ones(1, N(t)); prediction.H{t}.mean]' * state.bW.covariance(:, :, o) * [ones(1, N(t)); prediction.H{t}.mean]);
            end
        end
    end
    
    prediction.P = cell(1, T);
    for t = 1:T
        if N(t) > 0
            pos = 1 - normcdf((+state.parameters.margin - prediction.F{t}.mean) ./ prediction.F{t}.covariance);
            neg = normcdf((-state.parameters.margin - prediction.F{t}.mean) ./ prediction.F{t}.covariance);
            prediction.P{t} = pos ./ (pos + neg);
        end
    end
end