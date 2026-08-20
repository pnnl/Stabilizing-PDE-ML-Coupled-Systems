ENV["JULIA_CUDA_SILENT"] = true

using Flux
# using Flux: mse
# using Flux.Data: DataLoader
using Random
using Printf
using IterTools
using ProgressMeter: @showprogress
# using Flux: @epochs
# using OptimizationOptimisers

using BSON: @save
using BSON: @load

using Plots
using LinearAlgebra
using FFTW
using DelimitedFiles

# Solving viscous Burgers' by training an NN to approximate the first derivative, which
# is then differentiated to approximate the viscous term
# Training conducted using the exact solutions of VB
# Training conducted on Tsz number of sets with intermediate values also used for the loss
# calculation, chosen randomly in the time interval, for Ns number of ICs

include("MiscFuns.jl")

# Random.seed!(1234)

Nu = 2^8;

aa = [0 2*pi];
afac = 2*pi/(aa[2]-aa[1]);

xvUnif = range(aa[1],aa[2],Nu+1); xvUnif = xvUnif[1:end-1];
xvUnif = reshape(xvUnif,Nu,1);

D = 2; 
Wid = 2^9;

M_max = 24;
Ns = 50;
Ns_test = Int(2*Ns);

sig(x) = tanh.(x); lea_rt = 1.0e-4; 

num_epochs = 1000;
bsz = 100;

Tf = 1.0;
bst = 1.0e4;
Ts = ceil(bst*Tf); dt = Tf/Ts;

rr = 1;

tau = 10;
if rr == 1 
	Tsz = 250;
	Tvs_tr = Tf*rand(Tsz);
	Tvs_te = Tf*rand(Tsz);
else
	Tsz = Int(Ts/tau);
	Tvs_tr = dt*(0:tau:(Ts-tau));
	Tvs_te = dt*(0:tau:(Ts-tau));
end

nu = 0.1;

Dx = afac*im*[0:(Nu/2-1) ; 0 ; (-Nu/2+1):-1];

Nsp = (0:Nu/2); 
Nf = exp.(-36*(Nsp/(Nu/2)).^36);
Fi = [Nf[1:end-1] ; 0 ; Nf[end-1:-1:2]];

VBFun(t,u) = real(ifft( -0.5*Fi.*Dx.*fft(u.^2 , (1,))  + nu*(Dx.^2).*fft(u , (1,)),(1,)));


showind = Int(num_epochs/10);
opt =  Adam(lea_rt);

if D > 1
	neurons = cat((Nu,Wid),[(Wid,Wid) for i in 1:D-2],(Wid,Nu),dims = (1,1));
	actvs = cat([sig for i in 1:D-1],identity,dims = (1,1));
else
	neurons = [(Nu,Nu)];
	actvs = [(identity)];
end
 
layers = [Dense(neurons[i][1],neurons[i][2],actvs[i]) for i in 1:D];
Mo = Chain(layers...)|>f64;

Tchain = dt*(0:tau);

freqs = 0:M_max; freqs = reshape(freqs,1,M_max+1);
F0s(x) = [cos.(freqs.*x) sin.(freqs[2:end]'.*x)]; 

Rvs_train = (2*rand(2*M_max+1,Ns) .- 1)./[((freqs').^2 .+ 1) ; (freqs[2:end].^2 .+ 1)].^(2/2);
Rvs_test = (2*rand(2*M_max+1,Ns_test) .- 1)./[((freqs').^2 .+ 1) ; (freqs[2:end].^2 .+ 1)].^(2/2);

Xtrain = zeros(Nu,Int(Tsz*Ns)); Ytrain = zeros(Nu,tau,Int(Tsz*Ns));
for ll = 1:Ns
	
	u0f(x) = F0s(x)*Rvs_train[:,ll];
	
	for tt = 1:Tsz
		Uv0 , ~ = rungek4sys(VBFun,0,Tvs_tr[tt],dt,u0f(xvUnif));		
		UvalsEx , ~ = rungek4sys(VBFun,0,(tau*dt),dt,Uv0[:,end]);
		
		Xtrain[:,Int((ll-1)*Tsz+tt)] = UvalsEx[:,1];
		Ytrain[:,:,Int((ll-1)*Tsz+tt)] = UvalsEx[:,2:Int(tau+1)];
	end
end

Xtest = zeros(Nu,Int(Tsz*Ns_test)); Ytest = zeros(Nu,tau,Int(Tsz*Ns_test));
for ll = 1:Ns_test
		
	u0f(x) = F0s(x)*Rvs_test[:,ll];
	
	for tt = 1:Tsz
		Uv0 , ~ = rungek4sys(VBFun,0,Tvs_te[tt],dt,u0f(xvUnif));		
		UvalsEx , ~ = rungek4sys(VBFun,0,(tau*dt),dt,Uv0[:,end]);
		
		Xtest[:,Int((ll-1)*Tsz+tt)] = UvalsEx[:,1];
		Ytest[:,:,Int((ll-1)*Tsz+tt)] = UvalsEx[:,2:Int(tau+1)];
	end
end


function loss(x,y,Model)

	F(t,u) = real(ifft(Fi.*Dx.*fft(-0.5*u.^2 +  nu*Model(u) ,(1,)) ,(1,)));

	err_val = 0;
	uval = x;
	for tt = 1:tau
		uval = rungek4sys_one(F,0,dt,uval);
		err_val += sum((uval - y[:,tt,:]).^2); 
	end

	return err_val

end


par = Flux.params(Mo);

Mo1 = Chain(layers...)|>f64;
par1 = Flux.params(Mo1);
train_MSE1 = 1.0e10;

train_loader = Flux.DataLoader((Xtrain,Ytrain), batchsize=bsz); # Placing the training data into batches
Errs = zeros(num_epochs,2); 

@showprogress 1 "Training the model..." for ii = 1:num_epochs
	for (x,y) in train_loader
		Flux.train!((x, y) -> loss(x, y, Mo), par, [(x,y)], opt)
	end

	train_MSE0 = loss(Xtrain,Ytrain,Mo);
	test_MSE0 = loss(Xtest,Ytest,Mo);	
	
	Errs[ii,:] = [train_MSE0  test_MSE0];

	if train_MSE0 < train_MSE1
		for jj = 1:Int(2*D)
	 		global par1[jj] .= par[jj];
		end
	 	global train_MSE1 = train_MSE0;
	end

	if ii%showind == 0
		# opt.eta = lea_rt*(showind/ii); 
				
		println("\nTrain L2 Error $train_MSE0")
		println("Test L2 Error $test_MSE0")
	end

end

writedlm("MSerrs",Errs)

@save @sprintf("DD_%i_%i_%i_%i_%i_%i.bson",M_max,D,tau,Ns,Tsz,num_epochs) Mo1
plot(1:num_epochs,Errs,yaxis=:log)







