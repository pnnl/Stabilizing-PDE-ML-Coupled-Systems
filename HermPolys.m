function Y = HermPolys(X,d)

if d == 0
    Y = 1 + 0*X;
elseif d == 1
    Y = X;
elseif d == 2
    Y = (X.^2 - 1)/sqrt(2);
elseif d == 3
    Y = (X.^3 - 3*X)/sqrt(6);
elseif d == 4
    Y = (X.^4 - 6*X.^2 + 3)/sqrt(24);
elseif d == 5
    Y = (X.^5 - 10*X.^3 + 15*X)/sqrt(120);
elseif d == 6
    Y = (X.^6 - 15*X.^4 + 45*X.^2 - 15)/sqrt(720);
elseif d == 7
    Y = (X.^7 - 21*X.^5 + 105*X.^3 + 105*X)/sqrt(5040);
end




end