function [u,T] = rungek4sys(f,a,b,h,alpha)
m = length(alpha);
n = ceil((b-a)/h); h = (b-a)/n;
u = zeros(m,n+1);
T = linspace(a,b,n+1);
u(:,1) = alpha;
for i = 1:n
    %[i n]
    
    k1 = f(T(i),u(:,i));
    k2 = f(T(i)+h/2,u(:,i)+h*k1/2);
    k3 = f(T(i)+h/2,u(:,i)+h*k2/2);
    k4 = f(T(i)+h,u(:,i)+h*k3);
    u(:,i+1) = u(:,i) + h/6*(k1 + 2*k2+ 2*k3 + k4);
end


