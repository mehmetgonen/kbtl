% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbtl_supervised_classification_variational_test(K, state)
    T = length(K);
    N = zeros(T, 1);
    for t = 1:T
        N(t) = size(K{t}, 2);
    end

    for t = 1:T
        if N(t) > 0
            prediction.H{t}.mean = state.A{t}.mean' * K{t};
        end
    end

    prediction.f = cell(1, T);
    for t = 1:T
        if N(t) > 0
            prediction.f{t}.mean = [ones(1, N(t)); prediction.H{t}.mean]' * state.bw.mean;
            prediction.f{t}.covariance = 1 + diag([ones(1, N(t)); prediction.H{t}.mean]' * state.bw.covariance * [ones(1, N(t)); prediction.H{t}.mean]);
        end
    end
    
    prediction.p = cell(1, T);
    for t = 1:T
        if N(t) > 0
            pos = 1 - normcdf((+state.parameters.margin - prediction.f{t}.mean) ./ prediction.f{t}.covariance);
            neg = normcdf((-state.parameters.margin - prediction.f{t}.mean) ./ prediction.f{t}.covariance);
            prediction.p{t} = pos ./ (pos + neg);
        end
    end
end