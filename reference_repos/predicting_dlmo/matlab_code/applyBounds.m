function y = applyBounds(x)

y = mod(x,24);
y(y > 12) = y(y > 12) - 24;
y(y <= -12) = y(y <= -12) + 24;

end