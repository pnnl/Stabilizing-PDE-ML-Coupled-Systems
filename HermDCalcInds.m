function Y = HermDCalcInds(X,C)

% Uses a precomputed NCKenum, as opposed to HermDCalc

N = size(X,1);
Jval = size(C,1);

Y = zeros(Jval,size(X,2));

for ii = 1:Jval
    y = ones(1,size(X,2));
    for ll = 1:N
        y = y.*HermPolys(X(ll,:),C(ii,ll));
    end

    Y(ii,:) = y;
end


end