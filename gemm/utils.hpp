#pragma once

#include <random>
#include <cmath>

enum class InitVecFlag {
    Zero,
    One,
    IncreaseByOne,
    RandonValue,
};

void fill_array(float *v, int length, InitVecFlag flag) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (size_t i = 0; i < length; i++) {
        switch (flag)
        {
        case InitVecFlag::Zero : 
            v[i] = 0;
            break;
        case InitVecFlag::One : 
            v[i] = 1;
            break;
        case InitVecFlag::IncreaseByOne : 
            v[i] = i;
            break;
        case InitVecFlag::RandonValue :
            v[i] = dist(mt);
            break;
        default:
            printf("Error InitVecFlag value.\n");
            exit(1);
        }
    }
}

void display_matrix(float *a, int length, int lda) {
    int col_num = length / lda;
    int row_num = lda;
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            printf(" %7.2f", a[j * lda + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void compare_array(float *a, float *b, int size) {
    float diff = 0.0;
    for (int i = 0; i < size; i++) {
        diff = std::abs(a[i] - b[i]);
        if (diff > 1e-3) {
            printf("Check error: %.2f vs %.2f\n", a[i], b[i]);
            return;
        }
    }
    printf("Check pass.\n");
    return;
}
