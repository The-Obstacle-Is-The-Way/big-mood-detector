function drawDays(time,localHourAtStart)
boundValue = 5;
hourCounter = 0;
while hourCounter < max(time) + 3*24
    
    x = [hourCounter,hourCounter,hourCounter+1,hourCounter+1];
    y = [-boundValue,boundValue, boundValue, -boundValue];
    
    if mod(hourCounter + localHourAtStart,24) >= 8
        colorValue = [1.0, 1.0, 0.95];
    else
        colorValue = [0.5, 0.5, 0.7];
    end
    patch(x,y,colorValue,'EdgeColor',colorValue);
    
    hourCounter = hourCounter + 1;
end
box off;
set(gcf,'Color','w');
set(gcf,'Position', [440   415   853   383]); 
end