function [tc,y] = circadianModel(lightStruct,params,ics,model)
% CIRCADIANMODEL  Simulates the model of the circadian clock given light
% input in a struct, a value for tau, and initial conditions; assumes light sufficiently padded
% to remove initial condition effects
%
% [T,Y] = circadianModel(LIGHT,TAU) returns the timesteps in T and output
% variables (x, xc, n) in Y
%
% Example:
% [t,y] = circadianModel(struct('dur',dur,'time',time,'light',light_vec),24.2,[1,1,1]);

if strcmp(model,'simple')
    func = @simple;
elseif strcmp(model,'kronauerJewett')
    func = @kronauerJewett;
elseif strcmp(model,'nonphotic')
    func = @nonphotic;
elseif strcmp(model,'hannay_twopop')
    func = @hannay_twopop;
else
    func = @hannay;
end

dt = 1/60;
[tc,y] = ode23s(func, min(lightStruct.time):dt:lightStruct.dur,ics,[],lightStruct);

% As a sanity check, here's RK-4
% tc = 1/60:dt:lightStruct.dur;
% y = ode4(func,tc,ics,lightStruct);

    function dydt = simple(t,y,u)
        % Forger, 1999 
        x = y(1);
        xc = y(2);
        n = y(3);
        
        if(~isfield(u,'is_blocks'))
            I = interp1(u.time,u.light,t,'linear','extrap');
        else
            I = u.light(find(u.time < t, 1, 'last' ));
        end
        
        if(isempty(I))
            I = 0;
        end
        
        I0 = 9500;
        p = 0.6; 
        a0 = 0.05;
        alpha = a0.*(I.^p/I0.^p);
        tx = 24.2;
        G = 33.75; % changed from 19.875
        k = 0.55;
        mu = 0.23;
        b = 0.0075; % changed from 0.013
        
        Bh = G*(1-n)*alpha;
        B = Bh*(1 - .4*x)*(1 - .4*xc);
        
        dydt(1) = pi/12*(xc + B);
        dydt(2) = pi/12*(mu*(xc - 4*xc^3/3) - x*((24/(.99669*tx))^2 + k*B));
        dydt(3) = 60*(alpha*(1-n) - b*n);
        
        dydt = dydt';
    end

    function dydt = nonphotic(t,y,u)
        % St. Hilaire Model 2007
        x = y(1);
        xc = y(2);
        n = y(3);
        if(~isfield(u,'is_blocks'))
            I = interp1(u.time,u.light,t,'linear','extrap');
        else
            I = u.light(find(u.time < t, 1, 'last' ));
        end
        
        I0 = 9500;
        p = 0.5;
        alpha0 = 0.1;
        alpha = alpha0*((I/I0)^p)*(I/(I+100));
        
        G = 37;
        Bh = G*alpha*(1-n);
        B = Bh*(1-0.4*x)*(1-0.4*xc);
        
        beta =0.007;
        dydt(3) = 60*(alpha*(1-n) - beta*n);
        
        sigma = interp1(u.time,u.sw,t,'linear','extrap');
        if sigma < 1/2
            sigma = 0;
        else
            sigma = 1;
        end
        
        tx = 24.2;
        k = .55;
        mu = .1300;
        q = 1/3;
        rho = 0.032;
        
        C = mod(t,24);
        % C = atan(xc/x)*24/(2*pi);
        phi_xcx = -2.98;
        phi_ref = 0.97;
        CBTmin = phi_xcx + phi_ref;
        CBTmin = CBTmin*24/(2*pi);
        psi_cx = C - CBTmin;
        psi_cx = mod(psi_cx,24);
        
        Nsh = rho*(1/3 - sigma);
        if psi_cx > 16.5 && psi_cx < 21
            Nsh = rho*(1/3);
        end
        Ns = Nsh*(1 - tanh(10*x));
        
        dydt(1) = pi/12* (xc + mu*(1/3*x+4/3*x^3-256/105*x^7) + B + Ns);
        dydt(2) = pi/12* (q*B*xc - x*((24/(0.99729*tx))^2 + k*B));
        
        dydt = dydt';
    end

    function dydt = kronauerJewett(t,y,u)
        % Higher-order model
        x = y(1);
        xc = y(2);
        n = y(3);
      
        I = u.light(find(u.time < t, 1, 'last' ));
        
        if(isempty(I))
            I = 0;
        end
        
        I0 = 9500;
        p = .6;
        a0 = 0.16;
                
        if isKey(params,'alphaScalar')
             a0 = a0*params('alphaScalar');
        end
        
        alpha = a0*(I.^p/I0.^p);
        tx = 24.2;
        G = 19.875;
        b = 0.013;
        k = .55;
        
        mu = .1300;
        q = 1/3;
        Bh = G*alpha*(1-n);
        B = Bh*(1-0.4*x)*(1-0.4*xc);
        
        dydt(1) = pi/12* (xc + mu*(1/3*x+4/3*x^3-256/105*x^7) + B); 
        dydt(2) = pi/12* (q*B*xc - x*((24/(0.99729*tx))^2 + k*B));
        dydt(3) = 60*(alpha*(1-n) - b*n);
        
        dydt = dydt';
    end

    function dydt = hannay(t,y,u)
        
        R = y(1);
        psi = y(2);
        n = y(3);
        
        omeg0 = 0.263524; 
        K = 0.06358;
        gam = 0.024;
        Beta1 = -0.09318; 
        A1 = 0.3855;
        A2 = 0.1977;
        betaL1 = -0.0026; 
        betaL2 = -0.957756; 
        sigm = 0.0400692; 
        G = 33.75;
        alph0 = 0.05;
        delt = 0.0075;
        p = 1.5;
        I0 = 9325.0;
        
        light = @(t) u.light(find(u.time < t, 1, 'last' ));

         if(isempty(light(t)))
            light = @(t) 0;
         end
        
        if isKey(params,'alphaScalar')
            alph0 = alph0*params('alphaScalar');
        end
        
        alphaFunc = @(t) alph0*light(t)^p/(light(t)^p+I0);
        
        Bhat = G*(1-n)*alphaFunc(t); % added an equation for B(t)
        nEquation = 60.0*(alphaFunc(t)*(1-n)-delt*n);
        %lightAmp = A1*0.5*G*(1-n)*alphaFunc(t)*(1-R^4)*cos(psi+betaL1)+A2*0.5*G*(1-n)*alphaFunc(t)*R*(1-R^8)*cos(2*psi+betaL2);
        %lightPhase = sigm*G*(1-n)*alphaFunc(t)-A1*0.5*G*(1-n)*alphaFunc(t)*(R^3+1/R)*sin(psi+betaL1)-A2*0.5*G*(1-n)*alphaFunc(t)*(1+R^8)*sin(2*psi+betaL2);
        
        lightAmp = A1*0.5*Bhat*(1-R^4)*cos(psi+betaL1)+A2*0.5*Bhat*R*(1-R^8)*cos(2*psi+betaL2);
        lightPhase = sigm*Bhat-A1*0.5*Bhat*(R^3+1/R)*sin(psi+betaL1)-A2*0.5*Bhat*(1+R^8)*sin(2*psi+betaL2);
        
        %dRdt = -gam*R+K/2*R*(1-R^4)+lightAmp;
        %dPsidt = omeg0+lightPhase;
        dRdt = -gam*R+K/2*cos(Beta1)*R*(1-R^4)+lightAmp;
        dPsidt = omeg0+K/2*sin(Beta1)*(1+R^4)+lightPhase;
        dndt = nEquation;
        
        dydt = [dRdt; dPsidt; dndt];
        
    end

    function dydt = hannay_twopop(t,y,u) 
        
        Rv = y(1);
        Rd = y(2);
        psi_v = y(3);
        psi_d = y(4);
        n = y(5);
        
        tau_v = 24.5;
        tau_d = 24;
        K_vv = 0.05;
        K_dd = 0.04;
        K_dv = 0.01; 
        alphaRatio = 2;

        if isKey(params,'alphaRatio')
            alphaRatio = params('alphaRatio');
        end
        K_vd = alphaRatio*K_dv;
        
        gamma = 0.024;
        A1 = 0.440068;
        A2 = 0.159136;
        beta_l1 = 0.06452;
        beta_l2 = -1.38935;
        sigma = 0.0477375;
        G = 33.75;
        alph0 = 0.05;
        delt = 0.0075;
        p = 1.5;
        I0 = 9325.0;
        
        light = @(t) u.light(find(u.time < t, 1, 'last' ));
        
         if(isempty(light(t)))
            light = @(t) 0;
         end
                
        alphaFunc = @(t) alph0*light(t)^p/(light(t)^p+I0);
        Bhat = G*(1-n)*alphaFunc(t); 
        nEquation = 60.0*(alphaFunc(t)*(1-n)-delt*n);
        
        lightAmp = A1*0.5*Bhat*(1-Rv^4)*cos(psi_v+beta_l1)+A2*0.5*Bhat*Rv*(1-Rv^8)*cos(2*psi_v+beta_l2);
        lightPhase = sigma*Bhat-A1*0.5*Bhat*(Rv^3+1/Rv)*sin(psi_v+beta_l1)-A2*0.5*Bhat*(1+Rv^8)*sin(2*psi_v+beta_l2);
        
        dRvdt = -gamma*Rv+K_vv/2*Rv*(1-Rv^4)+K_dv/2*Rd*(1-Rv^4)*cos(psi_d-psi_v)+lightAmp;
        dRddt = -gamma*Rd+K_dd/2*Rd*(1-Rd^4)+K_vd/2*Rv*(1-Rd^4)*cos(psi_d-psi_v);
        dPsivdt = 2*pi/tau_v + K_dv/2*Rd*(1/Rv+Rv^3)*sin(psi_d-psi_v)+lightPhase;
        dPsiddt = 2*pi/tau_d - K_vd/2*Rv*(1/Rd+Rd^3)*sin(psi_d-psi_v);
        dndt = nEquation;
        
        dydt = [dRvdt; dRddt; dPsivdt; dPsiddt; dndt];
        
    end
end



