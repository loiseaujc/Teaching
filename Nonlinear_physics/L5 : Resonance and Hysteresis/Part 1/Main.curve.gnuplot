set table "Main.curve.table"; set format "%.5f"
set samples 25; plot [x=-5:5]  f(x,y) = y**2 + (x**2 - 5)*(4*x**4 - 20*x**2 + 25); set xrange [-4:4]; set yrange [-15:15]; set view 0,0; set isosample 1000,1000; set cont base; set cntrparam levels incre 0,0.1,0; unset surface; splot f(x,y) 
