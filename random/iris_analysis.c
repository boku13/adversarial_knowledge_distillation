#include <stdio.h>
#include <string.h>
#include <float.h>
#include "iris_analysis.h"

void readIrisData(const char* filename, 
                  double* sum_sepal_len, double* sum_sepal_wid,
                  double* sum_petal_len, double* sum_petal_wid,
                  double* max_sepal_len, double* min_petal_wid,
                  SpeciesData species_stats[]) {
    FILE* file = fopen(filename, "r");
    if (!file) return;

    *max_sepal_len = -DBL_MAX;
    *min_petal_wid = DBL_MAX;
    int setosa_idx = -1, versicolor_idx = -1, virginica_idx = -1;

    char line[200];
    while (fgets(line, sizeof(line), file)) {
        double sl, sw, pl, pw;
        char species[20];
        sscanf(line, "%lf,%lf,%lf,%lf,%19s", &sl, &sw, &pl, &pw, species);

        // Update global sums
        *sum_sepal_len += sl;
        *sum_sepal_wid += sw;
        *sum_petal_len += pl;
        *sum_petal_wid += pw;

        // Track max sepal length and min petal width
        if (sl > *max_sepal_len) *max_sepal_len = sl;
        if (pw < *min_petal_wid) *min_petal_wid = pw;

        // Update species-specific data for petal length
        int found = 0;
        for (int i = 0; i < 3; i++) {
            if (strcmp(species_stats[i].species, species) == 0) {
                species_stats[i].total_petal_length += pl;
                species_stats[i].count++;
                found = 1;
                break;
            }
        }
        if (!found) { // Initialize new species entry
            strcpy(species_stats[0].species, species);
            species_stats[0].total_petal_length = pl;
            species_stats[0].count = 1;
        }
    }
    fclose(file);
}

void computeSpeciesAvg(SpeciesData species_stats[], char* top_species) {
    double max_avg = -DBL_MAX;
    for (int i = 0; i < 3; i++) {
        if (species_stats[i].count == 0) continue;
        double avg = species_stats[i].total_petal_length / species_stats[i].count;
        if (avg > max_avg) {
            max_avg = avg;
            strcpy(top_species, species_stats[i].species);
        }
    }
}
