
#include <stdio.h>

int res = 0; 

void gp(int open, int close, int n){
    if(open > n || close > n){
        return;
    }

    if(open == n && close == n){
        res += 1;
        return;
    }

    if(open < n){
        gp(open + 1, close, n);
    }

    if(close < open){
        gp(open, close + 1, n);
    }
    return;
}



int main() {
    int n;
    scanf("%d", &n);
    gp(0, 0, n);
    printf("%d", res);
    return 0;
}
