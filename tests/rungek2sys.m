function [u,T] = rungek2sys(f,a,b,h,alpha)

% Explicit trapezoidal rule

m = length(alpha);
n = ceil((b-a)/h); h = (b-a)/n;
u = zeros(m,n+1);
T = linspace(a,b,n+1);
u(:,1) = alpha;
for i = 1:n
    %[i n]
    
    k1 = f(T(i),u(:,i));
    k2 = f(T(i)+h,u(:,i)+h*k1);
    
    u(:,i+1) = u(:,i) + h/2*(k1 + k2);
end


