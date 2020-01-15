function m = computeMatches(f1,f2)
pairwiseDist = dist2(f1, f2);
[n1, n2] = size(pairwiseDist);
m = zeros(1, n1);
for i = 1:n1,
    [d,ord] = sort(pairwiseDist(i,:));
    ratio = d(1)/d(2);
    if ratio < 0.8,
        m(i) = ord(1);
    end
end
function n2 = dist2(x, c)
[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Error')
end
n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
  ones(ndata, 1) * sum((c.^2)',1) - ...
  2.*(x*(c'));
if any(any(n2<0))
  n2(n2<0) = 0;
end