function [X,W] = gauss_herm_grid(N)

% Generates N point Gauss-Hermite quadrature rule with respect to w(x) =
% e^(-x^2/2)/sqrt(2*pi); X = [x_1,...,x_N] contains the nodes and W = [w_1,...,w_N]
% contains the weights

A = zeros(N);
for ii = 1:N-1
    bsq = sqrt(ii);
    
    A(ii,ii+1) = bsq;
    A(ii+1,ii) = bsq;
    
end

[V,D] = eig(A);

X = diag(D);

W = V(1,:)'; 
W = W.^2;


end