function [Kvs,Nrom,Ts_sim,Nq0,Jval,D] = KvsRread(file_ident)

fileID = fopen(file_ident);
V = fscanf(fileID,'%f');
fclose(fileID); 

Nrom = V(1);
Ts_sim = V(2);
Nq0 = V(3);
D = V(4);

Jval = nchoosek(2*Nrom-2+D,D);
Kvs = zeros(Jval,2*Nrom-2,Ts_sim+1);

kk = 4;

for tt = 1:Ts_sim+1
    Kvs(:,:,tt) = transpose(reshape(V(kk+1:kk+Jval*(2*Nrom-2)),2*Nrom-2,Jval));
    kk = kk + Jval*(2*Nrom-2);
end


if kk ~= length(V)
    error('incorrect read')
end

end