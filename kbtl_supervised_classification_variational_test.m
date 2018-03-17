function prediction = kbtl_supervised_classification_variational_test(K, state)
    T = length(K);
    N = zeros(T, 1);
    for t = 1:T
        N(t) = size(K{t}, 2);
    end

    for t = 1:T
        if N(t) > 0
            prediction.H{t}.mu = state.A{t}.mu' * K{t};
        end
    end

    prediction.f = cell(1, T);
    for t = 1:T
        if N(t) > 0
            prediction.f{t}.mu = [ones(1, N(t)); prediction.H{t}.mu]' * state.bw.mu;
            prediction.f{t}.sigma = 1 + diag([ones(1, N(t)); prediction.H{t}.mu]' * state.bw.sigma * [ones(1, N(t)); prediction.H{t}.mu]);
        end
    end
    
    prediction.p = cell(1, T);
    for t = 1:T
        if N(t) > 0
            pos = 1 - normcdf((+state.parameters.margin - prediction.f{t}.mu) ./ prediction.f{t}.sigma);
            neg = normcdf((-state.parameters.margin - prediction.f{t}.mu) ./ prediction.f{t}.sigma);
            prediction.p{t} = pos ./ (pos + neg);
        end
    end
end
