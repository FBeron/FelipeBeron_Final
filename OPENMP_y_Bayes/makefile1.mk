hits*.pdf : data*.dat
    python plot_gauss.py
    
hits*.pdf : plot_gauss.py
    python plot_gauss.py
    
data*.dat : Beron_gauss.c
    gcc -o BeronGauss_c -fopenmp Beron_gauss.c -lm
    ./BeronGauss_c