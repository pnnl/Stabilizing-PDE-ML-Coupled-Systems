clear
clc

% Calculates the memory kernels using the method described in https://arxiv.org/abs/2506.19274

tic

Nu = 2^8;
Nrom = 4;
Nq0 = 3;

D = 1;

aa = [0 2*pi];
afac = 2*pi/(aa(2) - aa(1));
nu = 0.1;


Tf = 20;
bst = 1.0e4;
Ts = ceil(bst*Tf); 
dt = Tf/Ts;

Tf_sim = Tf;
bst_sim = 1.0e3;
Ts_sim = ceil(bst_sim*Tf_sim); 
dt_sim = Tf_sim/Ts_sim;

jmp = bst/bst_sim;

Tvs = linspace(0,Tf,Ts+1);
Tvs_sim = linspace(0,Tf_sim,Ts_sim+1);

xvUnif = linspace(0,2*pi,Nu+1); xvUnif = xvUnif(1:end-1)';
h = (aa(2) - aa(1))/Nu;

Nsp = (0:Nu/2); 
Nf = exp(-36*(Nsp/(Nu/2)).^36);
Fi = [Nf(1:end-1) 0 Nf(end-1:-1:2)]';

Dx = 1i*afac*[0:Nu/2-1 0 -Nu/2+1:-1]';
Dx_red = 1i*afac*(1:Nrom-1)';

DownSamp = @(V) V(2:Nrom,:);
UpSamp = @(v) [zeros(1,size(v,2)) ; v ; zeros(Nu+1-2*Nrom,size(v,2)) ; conj(v(end:-1:1,:))];

F = @(t,a) -0.5*(Fi.*Dx.*(fft((ifft(a)).^2))) + nu*(Dx.^2.*a);
F0 = @(t,a) -0.5*(Dx.*(fft((ifft(a)).^2))) + nu*(Dx.^2.*a);
F_red = @(t,a) -0.5*(Dx_red.*DownSamp(fft(ifft(UpSamp(a)).^2))) + nu*(Dx_red.^2.*a);

Nqval = Nq0^(2*Nrom - 2);

Sigs = Nu*(exp(-(1:(Nrom-1)))');

[ICtmp , Ws] = CoeffGen(2*Nrom-2,Nq0);
Ctmp = [1 ; 1i];

ICs = zeros(Nqval,Nrom-1);

for jj = 1:Nrom-1
    ICs(:,jj) = ICtmp(:,2*jj-1:2*jj)*Ctmp*Sigs(jj);
end

Jval = nchoosek(2*Nrom+D-2,D);

HermAs = [zeros(Jval,1) NCKenum(1:(2*Nrom-2+D),2*Nrom-2)];
HermCs = HermAs(:,2:end) - HermAs(:,1:end-1) - 1;

Fvs = zeros(Jval,2*Nrom-2,Ts_sim+1);
Gvs = zeros(Jval,Jval,Ts_sim+1);

for ll = 1:Nqval
    
    [ll Nqval 1]

    a0_red = transpose(ICs(ll,:));
    a0 = UpSamp(a0_red);
    
    [Avals_ex,~] = rungek4sys(F,0,Tf,dt,a0); Avals_ex = Avals_ex(:,1:jmp:end);

    Rvs = F(0,Avals_ex);
    

    Mvs = DownSamp(Dx.*fft(ifft(Avals_ex).*ifft(Rvs)));
    Mvs_red = DownSamp(Dx.*fft(ifft(UpSamp(DownSamp(Avals_ex))).*ifft(UpSamp(DownSamp(Rvs)))));

    Mvs_diff = Mvs - Mvs_red;


    Fvs = Fvs - Ws(ll)*permute([real(Mvs_diff) ; imag(Mvs_diff)],[3,1,2])...
        .*HermDCalcInds([real(a0_red)./Sigs ; imag(a0_red)./Sigs],HermCs);


    Gvs = Gvs + Ws(ll)*permute(HermLiouvD(...
        [real(Avals_ex(2:Nrom,:))./Sigs ; imag(Avals_ex(2:Nrom,:))./Sigs],...
        [real(Rvs(2:Nrom,:))./Sigs ; imag(Rvs(2:Nrom,:))./Sigs],HermCs),[3,1,2])...
        .*HermDCalcInds([real(a0_red)./Sigs ; imag(a0_red)./Sigs],HermCs);

    if any(any(any(isnan(Fvs))))
        error('Fvs NANs')
        break
    end

end


Kvs = zeros(Jval,2*Nrom-2,Ts_sim+1);
Kvs(:,:,1) = Fvs(:,:,1);

Gmat = eye(Jval) + 0.5*dt_sim*Gvs(:,:,1);
Ginv = Gmat\eye(Jval);

for tt = 1:Ts_sim

    if mod(tt,100) == 0
        [tt Ts_sim 2]
    end

    Kvs(:,:,tt+1) = Ginv*(Fvs(:,:,tt+1) - 0.5*dt_sim*Gvs(:,:,tt+1)*Kvs(:,:,1) - ...
        dt_sim*tensorprod(Gvs(:,:,tt:-1:2),Kvs(:,:,2:tt),[2 3],[1 3]));

    if any(any(isnan(Kvs(:,:,tt+1))))
        error('Kvs NANs')
        break
    end

end

fname = ['KvsR_' num2str(Nu) '_' num2str(Nrom) '_' num2str(Nq0) '_' num2str(D)];
fileID = fopen(fname,'w');
fprintf(fileID,'%d\n%d\n%d\n%d\n\n',Nrom,Ts_sim,Nq0,D);

for tt = 1:Ts_sim+1
    for ii = 1:Jval
        for jj = 1:2*Nrom-2
            fprintf(fileID,'%.16e\t',Kvs(ii,jj,tt));
        end
        fprintf(fileID,'\n');
    end    
    fprintf(fileID,'\n\n');
end

fclose(fileID);


for ll = 1:Jval
    for kk = 1:2*Nrom-2

%         plot(Tvs_sim,reshape(Kvs(ll,kk,:),1,Ts_sim+1),'LineWidth',3)
        semilogy(Tvs_sim,abs(reshape(Kvs(ll,kk,:),1,Ts_sim+1)),'LineWidth',3)

        hold on
    end
end

hold off

grid on
set(gca,'fontsize',20)
set(gcf, 'Position',  [500, 300, 800, 700])
 
toc

function [A,Ws] = CoeffGen(N,D)

[L,W] = gauss_herm_grid(D);

if N == 1
    A = L;
    Ws = W;
else
    [B,Wb] = CoeffGen(N-1,D);
    
    A = zeros(D*size(B,1),size(B,2)+1);
    Ws = zeros(D*size(B,1),1);

    for dd = 1:D
        A((dd-1)*size(B,1)+1:dd*size(B,1),:) = [L(dd)*ones(D^(N-1),1) B];
        Ws((dd-1)*size(B,1)+1:dd*size(B,1)) = W(dd)*Wb;
    end 

end

end
