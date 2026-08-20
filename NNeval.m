function X = NNeval(Xin,L,Ws,bs,sig)

X = Xin;

for jj = 1:L
    Y = Ws{jj}*X + bs{jj};
    if jj < L
        X = sig(Y);
    else
        X = Y;
    end
end


end