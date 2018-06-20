#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>

int main()
{
    const unsigned int rowNum = 5000;
    const unsigned int colNum = 5000;
    const unsigned int data_arr_num = rowNum * colNum;
    const unsigned int data_arr_size = sizeof(float) * data_arr_num;
    unsigned int i, j;
    
    // Alloc the space for the data array
    float *data_arr = (float*)malloc(data_arr_size);
    if(!data_arr)
    {
        printf("Unable to assign the space for the original data arr!\n");
        return -1;
    }
    memset(data_arr, (float)0.0f, data_arr_size);
    
    // Random set the initial value to the array (0 ~ 10000.0f)
    srand((int)time(NULL));
    for(i = 0; i < data_arr_num; i++)
    {
        data_arr[i] = (float)(rand() * 10000.0f);
    }

    // Data processing
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    if(rowNum < 3 || colNum < 3)
    {
        printf("\nToo few data sample for the data processing!\n");
        free(data_arr);
        return 0;
    }
    else
    {
        for(i = 1; i < rowNum - 1; i++)
        {
            unsigned int presNum = i * colNum;
            for(j = 1; j < colNum - 1; j++)
            {
                data_arr[presNum + j] = (data_arr[presNum + j - colNum] + data_arr[presNum + j - 1] + 
                data_arr[presNum + j + 1] + data_arr[presNum + j + colNum]) / 4;
            }
        }
    }
    gettimeofday(&t2, NULL);
    printf("\nData avg processing done.\n%u ms elapsed.\n\n", ((unsigned int)t2.tv_usec - (unsigned int)t1.tv_usec) / 1000);
    return 0;
}