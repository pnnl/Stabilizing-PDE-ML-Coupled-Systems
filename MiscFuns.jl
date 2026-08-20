function rungek4sys_one(f,a,h,alpha)


k1 = f(a,alpha);
k2 = f(a+h/2,alpha+h*k1/2);
k3 = f(a+h/2,alpha+h*k2/2);
k4 = f(a+h,alpha+h*k3);

u = alpha + h/6*(k1 + 2*k2+ 2*k3 + k4);

return u

end

function rungek4sys_block(f,a,n,h,alpha)


for tt = 1:n 
	tval = a + (tt-1)*h;

	k1 = f(tval,alpha);
	k2 = f(tval+h/2,alpha+h*k1/2);
	k3 = f(tval+h/2,alpha+h*k2/2);
	k4 = f(tval+h,alpha+h*k3);

	alpha = alpha + h/6*(k1 + 2*k2+ 2*k3 + k4);
end

return alpha

end



function rungek4sys(f,a,b,h,alpha)

m = length(alpha);
n = Int(round((b-a)/h)); h = (b-a)/n;

u = zeros(m,n+1);
T = range(a,b,n+1);

u[:,1] .= alpha;

for ii = 1:n
    # [ii n]
    
    k1 = f(T[ii],u[:,ii]);
    k2 = f(T[ii]+h/2,u[:,ii]+h*k1/2);
    k3 = f(T[ii]+h/2,u[:,ii]+h*k2/2);
    k4 = f(T[ii]+h,u[:,ii]+h*k3);
    u[:,ii+1] = u[:,ii] + h/6*(k1 + 2*k2+ 2*k3 + k4);
end

return u,T

end


function rungek4sys_comp(f,a,b,h,alpha)

m = length(alpha);
n = Int(round((b-a)/h)); h = (b-a)/n;

u = zeros(m,n+1) + im*zeros(m,n+1);
T = range(a,b,n+1);

u[:,1] .= alpha;

for ii = 1:n
    # [ii n]
    
    k1 = f(T[ii],u[:,ii]);
    k2 = f(T[ii]+h/2,u[:,ii]+h*k1/2);
    k3 = f(T[ii]+h/2,u[:,ii]+h*k2/2);
    k4 = f(T[ii]+h,u[:,ii]+h*k3);
    u[:,ii+1] = u[:,ii] + h/6*(k1 + 2*k2+ 2*k3 + k4);
end

return u,T

end


function ViscBurgSolv(u0f,xv,Tvs,Dx,nu)

Nu = length(xv); 

SIs = [2:Int(Nu/2) ; Int(Nu/2)+2:Nu];

u0C = (sum(u0f(xv)))/Nu;
Uvs = zeros(Nu,length(Tvs));

HeatSolve(tv,vv) = ifft(exp.(tv.*(nu*(Dx).^2)).*fft(vv));

for tt = 1:length(Tvs)

    tval = Tvs[tt];
    u0unif = u0f(xv .- u0C*tval);
    
    Psik = fft(u0unif);
    
    Psik[1] = 0; Psik[Int(Nu/2+1)] = 0;
    Psik[SIs] = Psik[SIs]./Dx[SIs];
    hv = real(ifft(Psik)); 
    hv = hv .- hv[1];
    
    v0 = exp.(-hv/(2*nu));
    VVals = HeatSolve(tval,v0);
    
    WVals = -2*nu*log.(VVals);
    PsiExVals = real(ifft(Dx.*fft(WVals)));
    Uvs[:,tt] = PsiExVals .+ u0C;

end

return Uvs

end


function NCKenum(B,k)

n = length(B);

if k > n
    print("k is too large")
else
    A = zeros(binomial(n,k),k);
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
        C = NCKenum(B[(ii+1:end)'],k-1);
	
        
        A[kk+1:kk+size(C,1),:] = [B[ii]*ones(size(C,1),1) C];
        kk = kk + size(C,1);
    end

    A[kk+1,:] = B[n-k+1:end];
    kk = kk + 1;

end


if kk != binomial(n,k)
    print("wrong enumeration")
end

return A

end




