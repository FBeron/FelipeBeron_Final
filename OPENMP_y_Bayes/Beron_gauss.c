#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<omp.h>
#define PI 3.14159265358979323846264338327

int main(){
    #pragma omp parallel
    {
        int n_points;
        double x, y;
        double *list;
        long long i;
        FILE *out;
        char filename[128];
        double gauss_1;
    
        n_points = 1000;
        
        int thread_id = omp_get_thread_num();
        for(i=0;i<n_points;i++){
            x = drand48();
            y = drand48();
            gauss_1 = sqrt(-2.0 * log(x)) * cos(2.0* PI * y);
            list[i] = gauss_1;
        }
        sprintf(filename, "sample_%d.dat", thread_id);
        if(!(out = fopen(filename, "w"))){
        fprintf(stderr, "Problema abriendo el archivo\n");
        exit(1);
        }
        
        for(i=0;i<n_points;i++){
            fprintf(out, "%f\n", list[i]);
        }
        fclose(out);
        
    }
    return 0;
}