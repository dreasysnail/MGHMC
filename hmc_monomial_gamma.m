function [ samples ] = hmc_monomial_gamma( U, gradU, opt)
	
	% pass parameters
	dim = opt.dim;
	m = ones(dim,1)*opt.mass;
	x = ones(dim,1)*10000;
	a = opt.a;
	c = opt.c;
    
    samples = zeros(prod(dim),opt.numCollect);
	rej_count = 0;
	
	% booleans
	isSoft = opt.isSoft;
	
	% funs
	K_org = @(p) 1./(m(:))'*(abs(p(:)).^(1/a));
	g = @(p) sign(p).*K_org(p);
    if isSoft
        K_f = @(p) -g(p) + 2/c.*log(1+exp(c.*g(p)));
    else
        K_f = K_org;
    end      
	psi_f = @(p) 1- 2/(exp(c.*g(p)) + 1);
	opt.f1 = @(p) 1./m./a.*psi_f(p).*(abs(p).^(1/a-1));
	
	if isSoft
		while true
			p = (randg(a,size(x)).*m).^a.*sign(rand(size(x))-0.5);
            if rand(1)< exp(K_org(p)-K_f(p)); break; end;
		end
	else
		p = (randg(a,size(x)).*m).^a.*sign(rand(size(x))-0.5);
	end	
		 
	hitCounts = 0; totalCounts = 0;
	for i = 1:opt.numCollect+opt.numBurnin    
	    if opt.verbose
	        if mod(i,5000) == 0
	            disp([num2str(i) ' iterations completed.']);
	            disp(['Acceptance: ' num2str(1-rej_count/i) '  Wall hits ratio:  ' num2str(hitCounts/totalCounts)]);
	        end
        end
        if isSoft
            while true
                p = (randg(a,size(x)).*m).^a.*sign(rand(size(x))-0.5);
                if rand(1)< exp(K_org(p)-K_f(p)); break; end;
            end
        else
            p = (randg(a,size(x)).*m).^a.*sign(rand(size(x))-0.5);
        end	
	    oldEnergy = K_f(p) + U(x); 
        oldX = x;
	    totalCounts = totalCounts + length(p(:))*(opt.nstep+1);
	    if opt.isRM
	        [x,p,hitCounts] = leapfrog_rm(p,x,opt.dt,nstep,gradU,a, hitCounts,G,U);   
	    else
	        [x,p,hitCounts] = leapfrog(p,x,gradU,m,a,hitCounts,opt);
	    end

	    % M-H test
	    if opt.isMH 
	        newEnergy  =  K_f(p) + U(x);
	        if exp(oldEnergy- newEnergy) < rand(1)
	            rej_count = rej_count + 1;
	            x = oldX;
	        end
        end
        if i>opt.numBurnin
            samples(:,i-opt.numBurnin) = x(:);
        end
	end
end


function [x,p,hitCounts] = leapfrog(p,x,gradU,m,a,hitCounts,opt)
    %first half
    dt = opt.dt;
    dtt = dt.*2* rand(1);
	nstep = floor(opt.nstep*0.1 + rand(1)*opt.nstep*2);
	if opt.isSoft
		grad_k = opt.f1;
	else
		grad_k = @(p) sign(p)./(m*a).*(abs(p).^(1/a-1));
	end 
    if opt.isDiscrete
        assert(a==1);
    else
        pprev = p;
        p = p - gradU( x ) .* dtt/2;
        if (a>0.5||opt.isReflect) && any(p(:).*pprev(:)<0)
            ix = (p.*pprev<0);  p(ix) = -pprev(ix); 
            hitCounts = hitCounts + sum(p(:).*pprev(:)<0);
        end
    end
    for j = 1 : nstep-1
        pprev = p; xprev = x; dtt = dt.*2* rand(1,1);
        if opt.isDiscrete; dtt = m.*(dtt>1);end;
        if opt.printEnergy;  fprintf('(%d %f %f)',x,p,dtt); oldEnergy  =  1./(m(:))'*(abs(p(:)).^(1/a))  + U(x);disp(oldEnergy); end
        x = x + grad_k(p).* dtt;
        assert(~any(isnan(x(:))));
        if opt.isPosCon && any(x<0)
            x = xprev; p = -pprev; continue;
        end
        %x = x + p.*dtt;       
        %dtt = dt;
        if opt.isDiscrete
            p = p - (U( x )- U(xprev))*sign(p).*dtt; 
        else
            p = p - gradU( x ) .* dtt;
        end
        if (a>0.5||opt.isReflect) && any(p(:).*pprev(:)<0)
            %ix = (p.*pprev<0); x = xprev; p = pprev;  p(ix) = -p(ix); 
            ix = (p.*pprev<0); x(ix) = xprev(ix);  p(ix) = -pprev(ix); 
            hitCounts = hitCounts + sum(p(:).*pprev(:)<0);
            %[x,p] = reflect_1st(x,p,xprev,pprev,dtt,a,gradU,m);
            continue;
        end
%         if printEnergy; newEnergy  =  1./(m(:))'*(abs(p(:)).^(1/a))  + U(x);disp('new');disp(newEnergy); end
    end

    pprev = p; xprev = x; dtt = dt.*2* rand(1,1);
    if opt.isDiscrete;return;end;
    x = x + grad_k(p).* dtt;
    if opt.isPosCon && any(x<0)
        x = xprev; p = -pprev; 
    end
    p = p - gradU( x ) .* dtt/2;
    if (a>0.5||opt.isReflect) && any(p(:).*pprev(:)<0)
        ix = (p.*pprev<0);  x(ix) = xprev(ix);  p(ix) = -pprev(ix); 
        hitCounts = hitCounts + sum(p(:).*pprev(:)<0);
    end
end


function [x,p,hitCounts] = runge_kutta(p,x,dt,nstep,gradU,m,a, hitCounts,U)   
    %first half
    %dtt = dt;
    for j = 1 : nstep
        dtt = dt.*2* rand(1,1);
        pprev = p; xprev = x;
        k1x = ((p>0)*2-1)./(m*a).*(abs(p).^(1/a-1));
        k1p = - gradU( x );
        x = x + k1x.* dtt/2;
        p = p + k1p.* dtt/2;
        if a>=0.5 && any(p(:).*pprev(:)<0)
            ix = (p.*pprev<0); x(ix) = xprev(ix);  p(ix) = -pprev(ix); 
            hitCounts = hitCounts + sum(p(:).*pprev(:)<0);
            continue;
        end
        k2x = ((p>0)*2-1)./(m*a).*(abs(p).^(1/a-1));
        k2p = - gradU( x );
        x = x + k2x.* dtt/2;
        p = p + k2p.* dtt/2;
        if a>=0.5 && any(p(:).*pprev(:)<0)
            ix = (p.*pprev<0); x(ix) = xprev(ix);  p(ix) = -pprev(ix); 
            hitCounts = hitCounts + sum(p(:).*pprev(:)<0);
            continue;
        end
        k3x = ((p>0)*2-1)./(m*a).*(abs(p).^(1/a-1));
        k3p = - gradU( x );
        x = x + k2x.* dtt;
        p = p + k2p.* dtt;
        if a>=0.5 && any(p(:).*pprev(:)<0)
            ix = (p.*pprev<0); x(ix) = xprev(ix);  p(ix) = -pprev(ix); 
            hitCounts = hitCounts + sum(p(:).*pprev(:)<0);
            continue;
        end
        k4x = ((p>0)*2-1)./(m*a).*(abs(p).^(1/a-1));
        k4p = - gradU( x );
        
        x = xprev + (k1x + 2*k2x + 2*k3x + k4x).* dtt/6;
        p = pprev + (k1p + 2*k2p + 2*k3p + k4p).* dtt/6;
        if a>=0.5 && any(p(:).*pprev(:)<0)
            ix = (p.*pprev<0); x(ix) = xprev(ix);  p(ix) = -pprev(ix); 
            hitCounts = hitCounts + sum(p(:).*pprev(:)<0);
            continue;
        end
        newEnergy  =  1./(m(:))'*(abs(p(:)).^(1/a))  + U(x);disp(newEnergy);
    end
end




