hits*.pdf : data*.dat
    python plot_gauss.py
    
hits*.pdf : plot_gauss.py
    python plot_gauss.py
    
data*.dat : Beron_gauss.c
    gcc -fopenmp Beron_gauss.c -o BeronGauss_c
    ./BeronGauss_c