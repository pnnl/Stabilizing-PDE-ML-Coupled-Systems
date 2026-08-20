clear
clc

% Uses the stabilized system with Nrom = 4 and the calculated memory kernels
% Can use kernels corresponding to either linear or cubic projections

[Ws,bs,act_fun,L,Ds] = NNread('surrogate_example',0);

% [Kvs,Nrom0,Ts_sim0,Nq0,Jval,D] = KvsRread('KvsR_linear');
[Kvs,Nrom0,Ts_sim0,Nq0,Jval,D] = KvsRread('KvsR_cubic');

if act_fun == 'relu'
    sig = @(x) max(x,0); 
elseif act_fun == 'tanh'
    sig = @(x) tanh(x); 
end

Nu0 = Ds(1);
Nu = 2^10;
Nrom = Nrom0;

aa = [0 2*pi];
afac = 2*pi/(aa(2) - aa(1));
nu = 0.1;

mov = 0;

Tf = 2; % The provided kernels are calculated up to T = 20
bst = 1.0e4;
Ts = ceil(bst*Tf); 
dt = Tf/Ts;


Tf_sim = Tf;
bst_sim = 1.0e3;
Ts_sim = ceil(bst_sim*Tf_sim); 
dt_sim = Tf_sim/Ts_sim;

jmp = bst/bst_sim;


u0f = @(x) sin(x); ymin = -1.2; ymax = 1.2;
% u0f = @(x) exp(sin(x)); ymin = -1.5; ymax = 2;
% u0f = @(x) cos(2*sin(x)); ymin = -1.2; ymax = 1.2;


xvUnif = linspace(0,2*pi,Nu+1); xvUnif = xvUnif(1:end-1)';
h = (aa(2) - aa(1))/Nu;

Nsp = (0:Nu/2); 
Nf = exp(-36*(Nsp/(Nu/2)).^36);
Fi = [Nf(1:end-1) 0 Nf(end-1:-1:2)]';

Dx = 1i*afac*[0:Nu/2-1 0 -Nu/2+1:-1]';
Dx_def = 1i*afac*[0:Nrom-1 zeros(1,Nu+1-2*Nrom) -Nrom+1:-1]';
Dx_red = 1i*afac*(1:Nrom-1)';

DownSamp = @(V) V(2:Nrom,:);
DownSamp0 = @(V) V(2:Nrom,:)*(Nu/Nu0);

UpSamp = @(v) [zeros(1,size(v,2)) ; v ; zeros(Nu+1-2*Nrom,size(v,2)) ; conj(v(end:-1:1,:))];
UpSamp0 = @(v) [zeros(1,size(v,2)) ; v ; zeros(Nu0+1-2*Nrom,size(v,2)) ; conj(v(end:-1:1,:))]*(Nu0/Nu);


Sigs = Nu*(exp(-(1:(Nrom-1)))');

a0 = DownSamp(fft(u0f(xvUnif)));
u0unif = ifft(UpSamp(a0));

F_ex = @(t,a) -0.5*(Fi.*Dx.*(fft((ifft(a)).^2))) + nu*(Dx.^2.*a);

F_red = @(t,a) -0.5*(Dx_red.*DownSamp(fft(ifft(UpSamp(a)).^2))) + ...
    nu*Dx_red.*DownSamp0(fft(NNeval(ifft(UpSamp0(a)),L,Ws,bs,sig)));



[AexVals,Tvs] = rungek4sys(F_ex,0,Tf_sim,dt,UpSamp(a0));
AexVals = DownSamp(AexVals(:,1:jmp:jmp*Ts_sim+1));

[MarkVals,Tvs_sim] = rungek2sys(F_red,0,Tf_sim,dt_sim,a0);

Kvals = permute((Kvs(:,1:Nrom-1,:)+1i*Kvs(:,Nrom:2*Nrom-2,:)),[2 1 3]); 

PsiVals = zeros(Nrom-1,Ts_sim+1);
PsiVals(:,1) = a0;

HermAs = [zeros(Jval,1) NCKenum(1:(2*Nrom-2+D),2*Nrom-2)];
HermCs = HermAs(:,2:end) - HermAs(:,1:end-1) - 1;



for tt = 1:Ts_sim

    if mod(tt,100) == 0
        [tt Ts_sim 3]
    end

    tval = (tt-1)*dt_sim;

    facs0 = ones(1,tt); 
    facs1 = ones(1,tt+1);

    if tt > 1
        facs0(1) = 1/2; facs0(tt) = 1/2;
    else
        facs0(1) = 0;
    end
    facs1(1) = 1/2; facs1(tt+1) = 1/2;

    PsiHerms = HermDCalcInds([real(PsiVals(:,1:tt))./Sigs ; imag(PsiVals(:,1:tt))./Sigs],HermCs);

    k1 = F_red(tval,PsiVals(:,tt)) + ...
        dt_sim*tensorprod(Kvals(:,:,tt:-1:1),PsiHerms.*facs0,[2 3],[1 2]);
    
    Psi_tmp = PsiVals(:,tt)+dt_sim*k1;
    PsiH_next = HermDCalcInds([real(Psi_tmp)./Sigs ; imag(Psi_tmp)./Sigs],HermCs);


    k2 = F_red(tval+dt_sim,Psi_tmp) + ...
        dt_sim*tensorprod(Kvals(:,:,tt+1:-1:1),[PsiHerms PsiH_next].*facs1,[2 3],[1 2]);

    PsiVals(:,tt+1) = PsiVals(:,tt) + 0.5*dt_sim*(k1 + k2);

    if any(isnan(PsiVals(:,tt+1)))
        error('NANs')
        break
    end
end


diff_Psi = PsiVals - AexVals;
diff_Mark = MarkVals - AexVals;

Psi_diff_vals = ifft(UpSamp(diff_Psi));
Mark_diff_vals = ifft(UpSamp(diff_Mark));

Errs_Psi = [max(abs(Psi_diff_vals)) ; sqrt(h*sum(Psi_diff_vals.^2))];
Errs_Mark = [max(abs(Mark_diff_vals)) ; sqrt(h*sum(Mark_diff_vals.^2))];


if mov
    psval = 0;
    
    UPsiVals = ifft(UpSamp(PsiVals));
    UMarkVals = ifft(UpSamp(MarkVals));
    UexVals = ifft(UpSamp(AexVals));

    for tt = 1:ceil(bst_sim/10):Ts_sim+1

        tval = (tt-1)*dt_sim;
        
        plot(xvUnif,UPsiVals(:,tt),xvUnif,UMarkVals(:,tt),xvUnif,UexVals(:,tt),'--','LineWidth',3); 

        an = annotation('textbox',[0.7 0.8 0.1 0.1],'str',['$t = $ ' num2str(tval,'%.2f')],...
        'fontsize',36,'Interpreter','latex','LineWidth',2,'LineStyle','none'); 

        grid on
        axis([0 2*pi ymin ymax])

        set(gca,'fontsize',20)
        set(gcf, 'Position',  [500, 300, 800, 700])

        pause(psval)

        [tt-1 Ts_sim]
        [(tt-1)*dt_sim Tf_sim]

        if tval < Tf
            delete(an)
        end
    end
    
    
else
    
    plot(Tvs_sim,Errs_Mark(2,:),Tvs_sim,Errs_Psi(2,:),'LineWidth',4)
    
    xlabel('$t$','Interpreter','latex','fontsize',24);
    ylabel('$||\mathcal{P}u_{ex} - u||$','Interpreter','latex','fontsize',24);
    

    legend('Markovian only','Markovian with memory kernels','Interpreter','latex','fontsize',28)

    grid on
    set(gca,'fontsize',24)
    set(gcf, 'Position',  [500, 300, 800, 700])
end

[max(max(abs(Mark_diff_vals))) max(max(abs(Psi_diff_vals))) ; ...
sqrt(h*dt_sim*[sum(sum(Mark_diff_vals.^2)) sum(sum(Psi_diff_vals.^2))])]


