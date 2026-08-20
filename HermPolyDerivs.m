function Y = HermPolyDerivs(X,d)

if d == 0
    Y = 0*X;
else
    Y = sqrt(d)*HermPolys(X,d-1);
end



end