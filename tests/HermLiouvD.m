function Zval = HermLiouvD(Xval,Yval,C)

N = size(Xval,1);
M = size(Xval,2);

Jval = size(C,1);

Zval = zeros(Jval,M);

for kk = 1:Jval
    for ll = 1:N
        y = ones(1,M);
        for jj = [1:ll-1 ll+1:N]
            y = y.*HermPolys(Xval(jj,:),C(kk,jj));
        end

        Zval(kk,:) = Zval(kk,:) + y.*Yval(ll,:).*HermPolyDerivs(Xval(ll,:),C(kk,ll));
    end
end


end