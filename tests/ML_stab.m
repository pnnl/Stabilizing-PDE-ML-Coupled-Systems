clear
clc

% Uses the trained surrogate and applies a low-pass filter for
% stabilization
% Nrom = 128 corresponds to no filtration at all

[Ws,bs,act_fun,L,Ds] = NNread('surrogate_example',0);

if act_fun == 'relu'
    sig = @(x) max(x,0); 
elseif act_fun == 'tanh'
    sig = @(x) tanh(x); 
end

Nu0 = Ds(1);
Nu = 2^8;
Nrom = 13; % 4, 13, 64, 128;

aa = [0 2*pi];
afac = 2*pi/(aa(2) - aa(1));
nu = 0.1;

mov = 0;

Tf = 10;
bst = 1.0e4;
Ts = ceil(bst*Tf); dt = Tf/Ts;

u0f = @(x) sin(x); ymin = -1.2; ymax = 1.2;
% u0f = @(x) exp(sin(x)); ymin = 0; ymax = 3;
% u0f = @(x) cos(2*sin(x)); ymin = -1.2; ymax = 1.2;



xvUnif = linspace(0,2*pi,Nu+1); xvUnif = xvUnif(1:end-1)';
h = (aa(2) - aa(1))/Nu;

Nsp = (0:Nu/2); 
Nf = exp(-36*(Nsp/(Nu/2)).^36);
Fi = [Nf(1:end-1) 0 Nf(end-1:-1:2)]';

Fi_rom = Nf(2:Nrom)';

Dx = 1i*afac*[0:Nu/2-1 0 -Nu/2+1:-1]';
Dx_red = 1i*afac*(1:Nrom-1)';

DownSamp = @(V) V(2:Nrom,:);
DownSamp0 = @(V) V(2:Nrom,:)*(Nu/Nu0);

UpSamp = @(v) [zeros(1,size(v,2)) ; v ; zeros(Nu+1-2*Nrom,size(v,2)) ; conj(v(end:-1:1,:))];
UpSamp0 = @(v) [zeros(1,size(v,2)) ; v ; zeros(Nu0+1-2*Nrom,size(v,2)) ; conj(v(end:-1:1,:))]*(Nu0/Nu);


a0 = DownSamp(fft(u0f(xvUnif)));

F_ex = @(t,a) -0.5*(Fi.*Dx.*(fft((ifft(a)).^2))) + nu*(Dx.^2.*a);
F_red = @(t,a) -0.5*(Fi_rom.*Dx_red.*DownSamp(fft(ifft(UpSamp(a)).^2))) + ...
    nu*Fi_rom.*Dx_red.*DownSamp0(fft(NNeval(ifft(UpSamp0(a)),L,Ws,bs,sig)));


[AexVals,Tvs] = rungek4sys(F_ex,0,Tf,dt,UpSamp(a0));

[MarkVals,~] = rungek2sys(F_red,0,Tf,dt,a0);

% If measuring ||Pu_ex - \phi_M||
% diff_Mark = MarkVals - DownSamp(AexVals);
% Mark_diff_vals = ifft(UpSamp(diff_Mark));

% If measuring ||u_ex - \phi_M||
diff_Mark = UpSamp(MarkVals) - AexVals;
Mark_diff_vals = ifft(diff_Mark);

Errs_Mark = [max(abs(Mark_diff_vals)) ; sqrt(h*sum(Mark_diff_vals.^2))];


if mov
    psval = 0;
    
    Uvals = ifft(UpSamp(MarkVals));
    UexVals = ifft(UpSamp(DownSamp(AexVals)));

    for tt = 1:ceil(bst/10):Ts+1

        tval = (tt-1)*dt;
        
        uex = UexVals(:,tt);
        plot(xvUnif,Uvals(:,tt),xvUnif,uex,'--','LineWidth',4); 

        an = annotation('textbox',[0.7 0.8 0.1 0.1],'str',['$t = $ ' num2str(tval,'%.2f')],...
        'fontsize',36,'Interpreter','latex','LineWidth',2,'LineStyle','none'); 

        grid on
        axis([0 2*pi ymin ymax])

        set(gca,'fontsize',24)
        set(gcf, 'Position',  [500, 300, 800, 700])

        pause(psval)

        [tt-1 Ts]
        [(tt-1)*dt Tf]

        if tval < Tf
            delete(an)
        end
    end
    
    
else
    
    plot(Tvs,Errs_Mark(2,:),'LineWidth',4)
    
    xlabel('$t$','Interpreter','latex','fontsize',24);
    ylabel('$|| \phi_{ex} - \phi||$','Interpreter','latex','fontsize',24);
    

    grid on
    set(gca,'fontsize',24)
    set(gcf, 'Position',  [500, 300, 800, 700])
end

Errs_Mark(:,end)



