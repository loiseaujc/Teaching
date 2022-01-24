set isosample 100,100
set ticslevel 0
set xlabel "x"
set ylabel "p1"
set zlabel "p2"
set xrange [-0.8:0.8]
set yrange [-0.5:0.5]
set zrange [-0.5:0.5]
# set view 300, 140
splot 4*x**3 + 2*u*x^2 + v*x, u, v
