
#include <stdio.h>
#include "iris_analysis.h"

int main() {
    char filename[100];
    scanf("%99s", filename);

    double sum_sepal_len = 0, sum_sepal_wid = 0, sum_petal_len = 0, sum_petal_wid = 0;
    double max_sepal_len = -1, min_petal_wid = -1;
    SpeciesData species_stats[3] = {0};

    readIrisData(filename, &sum_sepal_len, &sum_sepal_wid, &sum_petal_len, &sum_petal_wid,
                 &max_sepal_len, &min_petal_wid, species_stats);

    int total_entries = 0;
    for (int i = 0; i < 3; i++) total_entries += species_stats[i].count;

    if (total_entries == 0) {
        printf("No data found.\n");
        return 0;
    }

    // Compute averages
    double avg_sepal_len = sum_sepal_len / total_entries;
    double avg_sepal_wid = sum_sepal_wid / total_entries;
    double avg_petal_len = sum_petal_len / total_entries;
    double avg_petal_wid = sum_petal_wid / total_entries;

    // Find species with highest avg petal length
    char top_species[20] = "";
    computeSpeciesAvg(species_stats, top_species);

    // Print results
    printf("%.2lf %.2lf %.2lf %.2lf %.2lf %.2lf %s\n",
           avg_sepal_len, avg_sepal_wid, avg_petal_len, avg_petal_wid,
           max_sepal_len, min_petal_wid, top_species);

    return 0;
}
