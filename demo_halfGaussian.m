opt.nstep = 50; % leapfrog steps
opt.dt = 0.05; % stepsize
opt.dim = 1; % dimension
opt.numBurnin = 10000; 
opt.numCollect = 30000;

opt.isPosCon = false; % positive constraint domain
opt.isReflect = true; % force reflection
opt.isDiscrete = false; % discrete sampler
opt.printEnergy = false;
opt.verbose = true;
opt.isMH = true; % Metropolis Hasting
opt.isRM = false; % Riemannian manifold

% soften 
opt.isSoft = true; % soften kinetic
opt.c = 5; 

% set random seed
randn('seed',10);

% set potential and gradient
a0 = 1;
U = @(x) a0*x.^2;
gradUPerfect =  @(x) 2*x*a0 ;xGrid = [-3:xStep:3]; 

a = [2];
mass = [0.1];
for i = length(a)
    opt.mass = mass(i);
    opt.a = a(i);

    %% compare different HMC approaches
    figure(i);
    %% draw probability diagram 
    y = exp(- U(xGrid) );
    y = y / sum(y) / xStep;
    plot(xGrid,y);
    hold on;

    [samples] = hmc_monomial_gamma( U, gradUPerfect, opt);
    [yhmc,xhmc] = hist(samples, xGrid);
    yhmc = yhmc / sum(yhmc) / xStep;
    plot( xhmc, yhmc);
    legend( 'True Distribution',['MGHMC a=',num2str(opt.a)]);
    [ESS,g] = mcmc_ess_acorr(samples);corr = g(2)/g(1);
    disp(['a=',num2str(opt.a),'   ESS:',num2str(ESS),'      Rho(1):',num2str(corr)]);
end

