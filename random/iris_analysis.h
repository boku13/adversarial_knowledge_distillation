#ifndef IRIS_ANALYSIS_H
#define IRIS_ANALYSIS_H

typedef struct {
    char species[20];
    double total_petal_length;
    int count;
} SpeciesData;

void readIrisData(const char* filename, 
                  double* sum_sepal_len, double* sum_sepal_wid,
                  double* sum_petal_len, double* sum_petal_wid,
                  double* max_sepal_len, double* min_petal_wid,
                  SpeciesData species_stats[]);
void computeSpeciesAvg(SpeciesData species_stats[], char* top_species);

#endif
