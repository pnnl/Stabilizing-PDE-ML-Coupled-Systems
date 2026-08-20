function A = NCKenum(B,k)


n = length(B);

if k > n
    error('k is too large')
else
    A = zeros(nchoosek(n,k),k);
end


if k == n
    A = B;
    kk = 1;

elseif k == 0
    A = zeros(n,0);
    kk = 1;

elseif k == 1

    A = B';
    kk = n;

else

   kk = 0;

    for ii = 1:n-k
        C = NCKenum(B(ii+1:end),k-1);
        
        A(kk+1:kk+size(C,1),:) = [B(ii)*ones(size(C,1),1) C];
        kk = kk + size(C,1);
    end

    A(kk+1:kk+1,:) = B(n-k+1:end);
    kk = kk + 1;

end


if kk ~= nchoosek(n,k)
    error('wrong enumeration')
end


end



