function [Ws,bs,act_fun,L,Ds,Ns,Xc,alpha_NTK,alpha_CK,beta_CK] = NNread(file_ident,fl)

% fl for reading kernel data

fileID = fopen(file_ident);
V = fscanf(fileID,'%f');
fclose(fileID); 

if V(1) == 1.5
    act_fun = 'relu';
elseif V(1) == -1.5
    act_fun = 'tanh';
else
    error('activation function not read');
end

L = V(2);
Ds = V(3:3+L)';

k = L + 3;

Ws = cell(1,L);
bs = cell(1,L);

for jj = 1:L
    Ws{jj} = zeros(Ds(jj+1),Ds(jj));
    
    for ll = 1:Ds(jj)
        Ws{jj}(:,ll) = V(k+1:k+Ds(jj+1));
        k = k + Ds(jj+1);
    end

    bs{jj} = V(k+1:k+Ds(jj+1));
    k = k + Ds(jj+1);
end

if fl
    Ns = V(k+1);
    k = k + 1;

    Xc = V(k+1:k+2*Ns);
    k = k + 2*Ns;

    Xc = [Xc(1:Ns) Xc(Ns+1:2*Ns)]';

    alpha_NTK = V(k+1:k+Ns);
    k = k + Ns;

    alpha_CK = V(k+1:k+Ns);
    k = k + Ns;

    beta_CK = V(k+1:k+Ds(end-1)+1);
    k = k + Ds(end-1) + 1;

else
    Ns = 0;
    Xc = 0;
    alpha_NTK = 0;
    alpha_CK = 0;
    beta_CK = 0;
end


if k ~= length(V)
    error('incorrect read')
end

end