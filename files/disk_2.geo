// Gmsh project created on Tue Nov 05 15:59:01 2024
//+
radius = DefineNumber[ 10, Name "Parameters/radius" ];
//+
Point(1) = {radius, 0, 0, 1.0};
//+
Point(2) = {-radius, 0, 0, 1.0};
//+
Point(3) = {0, radius, 0, 1.0};
//+
Point(4) = {0, -radius, 0, 1.0};
//+
Point(5) = {0, 0, 0, 1.0};
//+
Circle(1) = {2, 5, 3};
//+
Circle(2) = {1, 5, 3};
//+
Circle(3) = {1, 5, 4};
//+
Circle(4) = {4, 5, 2};
//+
Curve Loop(1) = {2, -1, -4, -3};
//+
Plane Surface(1) = {1};
//+
Physical Surface(5) = {1};
//+
Physical Curve(6) = {2, 1, 4, 3};
//+
Transfinite Curve {2} = 100 Using Progression 1;
//+
Transfinite Curve {1} = 100 Using Progression 1;
//+
Transfinite Curve {4} = 100 Using Progression 1;
//+
Transfinite Curve {3} = 100 Using Progression 1;
