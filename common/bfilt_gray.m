%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%灰度图双边滤波%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = bfilt_gray(f,r,a,b)
% f灰度图；r滤波半径；a全局方差；b局部方差
[x,y] = meshgrid(-r:r);
w1 = exp(-(x.^2+y.^2)/(2*a^2));
f = tofloat(f);%f = im2double(f);
 
h = waitbar(0,'Applying bilateral filter...');
set(h,'Name','Bilateral Filter Progress');
 
[m,n] = size(f);
f_temp = padarray(f,[r r],'symmetric');
g = zeros(m,n);
for i = r+1:m+r
    for j = r+1:n+r
        temp = f_temp(i-r:i+r,j-r:j+r);
        w2 = exp(-(temp-f(i-r,j-r)).^2/(2*b^2));
        w = w1.*w2;
        s = temp.*w;
        g(i-r,j-r) = sum(s(:))/sum(w(:));
    end
    waitbar((i-r)/m);
end
% g = revertclass(g);