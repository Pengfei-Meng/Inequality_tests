% find the tangential point between surface and plane
% 
close all
clear all
clc

x_sol = [0.4897
         0.4897
         0.4535]; 

% plot 1     
[X,Y] = meshgrid(-1.2:.1:1.2);
Z = X.^2 + Y.^2; 

figure(1)
axis equal
surf(X, Y, Z)
alpha 0.9
hold on

% plot 3D point x_sol
scatter3(x_sol(1), x_sol(2), x_sol(3),'filled')
hold on

% plot 2 
[x,y,z] = sphere();
r = sqrt(2);
x = x(end/2+1:end, :); 
y = y(end/2+1:end, :); 
z = z(end/2+1:end, :); 
surf( r*x, r*y, r*z ) % sphere with radius 5 centred at (0,0,0) 
alpha 0.5
hold on

% x + y + z = 1.4329; 
point = x_sol'; 
normal = [1,1,1];

%# a plane is a*x+b*y+c*z+d=0
%# [a,b,c] is the normal. Thus, we have to calculate
%# d and we're set
d = -point*normal'; %'# dot product for less typing

%# create x,y
[xx,yy]=ndgrid(-1.5:1.5,-1.5:1.5);

%# calculate corresponding z
z = (-normal(1)*xx - normal(2)*yy - d)/normal(3);

%# plot the surface
surf(xx,yy,z)
alpha 0.3
xlabel('X')
ylabel('Y')
zlabel('Z')


