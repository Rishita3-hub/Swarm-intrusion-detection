%% ===============================================================
% FILE: uav_drone.m
% Rename to:
% uav_model.m
%
% PROFESSIONAL UAV DRONE MODEL
%
% After saving, change main file call from:
% uav_drone(...)
% to
% uav_model(...)
%% ===============================================================

function uav_model(px,py,pz,scale,color)

if nargin < 4
    scale = 1;
end

if nargin < 5
    color = [0 0.45 0.95];
end

hold on

%% ===============================================================
% BODY
%% ===============================================================
[Xs,Ys,Zs] = sphere(20);

surf(0.95*scale*Xs + px,...
     0.45*scale*Ys + py,...
     0.30*scale*Zs + pz,...
     'FaceColor',[0.10 0.10 0.10],...
     'EdgeColor','none',...
     'FaceLighting','gouraud');

%% ===============================================================
% TOP SHELL
%% ===============================================================
surf(0.52*scale*Xs + px,...
     0.22*scale*Ys + py,...
     0.13*scale*Zs + pz + 0.24*scale,...
     'FaceColor',color,...
     'EdgeColor','none');
material shiny
%% ===============================================================
% ARMS
%% ===============================================================
armL = 2.25*scale;
rad  = 0.08*scale;

draw_cylinder(px-armL,py,pz, px+armL,py,pz, rad,color);
draw_cylinder(px,py-armL,pz, px,py+armL,pz, rad,color);

%% ===============================================================
% ROTORS
%% ===============================================================
draw_rotor(px+armL,py,pz,scale);
draw_rotor(px-armL,py,pz,scale);
draw_rotor(px,py+armL,pz,scale);
draw_rotor(px,py-armL,pz,scale);

%% ===============================================================
% LANDING GEAR
%% ===============================================================
leg = 1.0*scale;

plot3([px-0.80 px-0.80],[py py],[pz-leg pz],...
    'Color',[0.18 0.18 0.18],...
    'LineWidth',2.4);

plot3([px+0.80 px+0.80],[py py],[pz-leg pz],...
    'Color',[0.18 0.18 0.18],...
    'LineWidth',2.4);

plot3([px-1.25 px+1.25],[py py],[pz-leg pz-leg],...
    'Color',[0.18 0.18 0.18],...
    'LineWidth',2.4);

%% ===============================================================
% CAMERA MODULE
%% ===============================================================
[Xc,Yc,Zc] = sphere(12);

surf(0.18*scale*Xc + px + 0.82*scale,...
     0.12*scale*Yc + py,...
     0.10*scale*Zc + pz - 0.12*scale,...
     'FaceColor',[0 0 0.95],...
     'EdgeColor','none');

%% ===============================================================
% GPS ANTENNA
%% ===============================================================
plot3([px px],[py py],[pz+0.30 pz+0.78],...
    'w','LineWidth',1.6);

plot3(px,py,pz+0.82,...
    'wo',...
    'MarkerFaceColor','w',...
    'MarkerSize',4);

%% ===============================================================
% STATUS LED
%% ===============================================================
plot3(px-0.25,py,pz+0.28,...
    'go',...
    'MarkerFaceColor','g',...
    'MarkerSize',4);

end

%% ===============================================================
% ROTOR
%% ===============================================================
function draw_rotor(x,y,z,s)

r = 0.65*s;
th = linspace(0,2*pi,60);

fill3(x+r*cos(th),...
      y+r*sin(th),...
      z*ones(size(th)),...
      [0.55 0.55 0.55],...
      'FaceAlpha',0.15,...
      'EdgeColor','none');

plot3([x-r x+r],[y y],[z z],...
    'k','LineWidth',1.5);

plot3([x x],[y-r y+r],[z z],...
    'k','LineWidth',1.5);

[Xh,Yh,Zh] = sphere(10);

surf(0.12*s*Xh+x,...
     0.12*s*Yh+y,...
     0.07*s*Zh+z,...
     'FaceColor',[0.18 0.18 0.18],...
     'EdgeColor','none');

end

%% ===============================================================
% CYLINDER
%% ===============================================================
function draw_cylinder(x1,y1,z1,x2,y2,z2,r,color)

n = 18;
th = linspace(0,2*pi,n);
zz = [0 1];

[TH,ZZ] = meshgrid(th,zz);

X = r*cos(TH);
Y = r*sin(TH);
Z = ZZ;

v = [x2-x1 y2-y1 z2-z1];
L = norm(v);

if L < 1e-8
    return
end

v = v/L;

k = cross([0 0 1],v);
s = norm(k);
c = dot([0 0 1],v);

if s < 1e-8

    R = eye(3);

else

    K = [0 -k(3) k(2);
         k(3) 0 -k(1);
        -k(2) k(1) 0];

    R = eye(3) + K + K*K*((1-c)/(s^2));

end

Z = Z*L;

for i = 1:numel(X)

    p = R*[X(i);Y(i);Z(i)];

    X(i)=p(1)+x1;
    Y(i)=p(2)+y1;
    Z(i)=p(3)+z1;

end

surf(X,Y,Z,...
    'FaceColor',color,...
    'EdgeColor','none');

end