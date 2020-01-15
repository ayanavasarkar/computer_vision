function plotfun

v = linspace(0, 1, 100);
[xx, yy] = meshgrid(v,v);

z = xx.^2.*yy.^2;

surf(xx, yy, z);

