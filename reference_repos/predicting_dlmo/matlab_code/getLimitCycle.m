function limit_cycle = getLimitCycle(wake, bed, max_light, model,params)
dt = 0.01;
time = 0:dt:(24-dt);
u = zeros(size(time));

for i = 1:length(time)
    if wake < bed
        if time(i) >= wake && time(i) < bed
            u(i) = max_light;
        end
    else
        if time(i) <= wake || time(i) > bed
            u(i) = max_light;
        end
    end
end

convergence_diff = 1e5;
ics = rand(1,3);
ics(1:2) = ics(1:2)*2 - 1;

while convergence_diff > 1e-3
    lightStruct = struct('dur',24,'time',time,'light',u);
    [tc,y] = circadianModel(lightStruct,params,ics,model);
    
    if(strcmp(model,'hannay'))
    	convergence_diff = norm(y(end,1) - ics(1)) + norm(mod(y(end,2) - ics(2) + pi,2*pi) - pi);
    else
        convergence_diff = norm(y(end,1:2) - ics(1:2));
    end
    
    ics = y(end,:);
    
    % figure(123); plot(tc,y(:,1)); hold on; drawnow; % Uncomment for sanity check
end

limit_cycle = @(t) interp1(tc,y,mod(t,24));

end