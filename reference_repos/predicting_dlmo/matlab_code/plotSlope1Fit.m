function plotSlope1Fit(predicted_phase, dlmo_min_mod24, stringInput)

figure()
x = dlmo_min_mod24;
y = predicted_phase;

y_shift = y;

for i = 1:length(y)
    if abs(x(i) + 24 - y(i)) < abs(x(i) - y(i)) && abs(x(i) + 24 - y(i)) < abs(x(i) - 24 - y(i))
        y_shift(i) = y(i) - 24;
    elseif abs(x(i) - 24 - y(i)) < abs(x(i) - y(i)) && abs(x(i) - 24 - y(i)) < abs(x(i) + 24 - y(i))
        y_shift(i) = y(i) + 24;
    end
end

sz = 100;
scatter(x,y_shift,sz,'MarkerEdgeColor',[0, 0, 0]);
hold on
xx = linspace(-24,24,100);

str = '#291B4F';
color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;

plot(xx,xx,'LineWidth',2,'Color',color);
box off
set(gcf,'color','w');

str = '#FCD42B';
color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
%plot(xx,xx+2,'--','LineWidth',2,'Color',[0, 0, 0]);
%plot(xx,xx-2,'--','LineWidth',2,'Color',[0, 0, 0]);
v1 = [-27, -23, 32, 28];
v2 = [-23, -27, 28, 32];

wantPatch = 0;
if wantPatch
    h = patch(v1,v2,[0.75,0.75,0.75], 'FaceAlpha', 0.5)
    set(h,'EdgeColor','none')
end

xlim([-24 24]); ylim([-24 24]);
R = corrcoef(x,y_shift);
R = R(1,2);
rhoc = 2*R*std(x)*std(y_shift)/(var(x) + var(y_shift) + (mean(x) - mean(y_shift))^2);
concordanceString = ['\rho_c =' num2str(rhoc)];

ylabel([stringInput '(hr)']); xlabel('Observed DLMO (hr)');

set(gca,'FontSize',18);set(gca,'LineWidth',2);
text(10,-4,concordanceString, 'FontSize',20)
axis([-18 18 -18 18])



