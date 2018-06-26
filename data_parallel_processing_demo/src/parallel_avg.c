#pragma warning (disable: 4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

typedef float my_type;

int main()
{
    // Create two vectors
    unsigned int i, j;
    const unsigned int rowNum = 10000;
    const unsigned int colNum = 10000;
    const unsigned int data_arr_num = rowNum * colNum;
    const unsigned int data_arr_size = data_arr_num * sizeof(my_type);

    my_type *org_data_list = (my_type*)malloc(data_arr_size);
    my_type *new_data_list = (my_type*)malloc(data_arr_size);
    unsigned int arr_info[2] = { rowNum, colNum };

    memset(org_data_list, 0, data_arr_size);
    memset(new_data_list, 0, data_arr_size);

    // Random set the number of the list
    srand((int)time(NULL));
    for(i = 0; i < data_arr_num; i++)
    {
        org_data_list[i] = (my_type)(((float)rand() / RAND_MAX) * 10000.0f);
    }

    // Set time struct
    struct timeval t1, t2;

    // Load the kernel codes into source_str
    FILE *fp_kernel = NULL;
    char *source_str = NULL;
    size_t source_size = 0;

    fp_kernel = fopen("src/parallel_avg_cl_kernel.cl", "r");
    if(!fp_kernel)
    {
        fprintf(stderr, "Failed to load kernel!\n");
        exit(1);
    }
    fseek(fp_kernel, 0, SEEK_END);
    source_size = ftell(fp_kernel);
    fseek(fp_kernel, 0, SEEK_SET);

    source_str = (char*)malloc(source_size);
    if(!source_str)
    {
        printf("Unable to assign the space for the kernel src buffer!\n");
        fclose(fp_kernel);
        exit(1);
    }
    fread(source_str, 1, source_size, fp_kernel);
    fclose(fp_kernel);

    // Obtain the platform and devices information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platform;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platform);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("%X\n", ret);
    printf("%s\n", source_str);

    // Create context for OpenCL
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    // Create command queue 
    // Here do not use the old version "clCreateCommandQueue" which has been deprecated before CL 2.0
    // Use "clCreateCommandQueueWithProperties" instead!
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    // Create buffer for each vector on the device
    cl_mem org_arr_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, data_arr_size, NULL, &ret);
    cl_mem arr_info_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(arr_info), NULL, &ret);
    cl_mem new_arr_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_arr_size, NULL, &ret);

    // Copy list_a and list_b into the GPU memory
    ret = clEnqueueWriteBuffer(command_queue, org_arr_mem, CL_TRUE, 0, data_arr_size, org_data_list, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, arr_info_mem, CL_TRUE, 0, sizeof(arr_info), arr_info, 0, NULL, NULL);

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    // Construct program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    // Create OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "avg_processing", &ret);

    // Set kernel parameters
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&org_arr_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&arr_info_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&new_arr_mem);

    // Execute the kernel
    size_t global_item_size = data_arr_num;    // Handle the whole list
    size_t local_item_size = 64;            // Cut into 64 pieces
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    gettimeofday(&t1, NULL);
    // Read ram buffer C into the local buffer C
    ret = clEnqueueReadBuffer(command_queue, new_arr_mem, CL_TRUE, 0, data_arr_size, new_data_list, 0, NULL, NULL);
    gettimeofday(&t2, NULL);
    // Write result into a file
    FILE *fp_result = NULL;
    char szBuf[128] = {0};

    fp_result = fopen("dep/result.txt", "w");
    if(!fp_result)
    {
        fprintf(stderr, "Unable to create an output file!\n");
        exit(2);
    }
    for(i = 0; i < rowNum; i++)
    {
        for(j = 0; j < colNum; j++)
        {
            memset(szBuf, 0, 128);
            sprintf(szBuf, "%f \t", new_data_list[i * colNum + j]);
            fwrite(szBuf, strlen(szBuf), 1, fp_result);
        }
        memset(szBuf, 0, 128);
        sprintf(szBuf, "\n");
        fwrite(szBuf, strlen(szBuf), 1, fp_result);
    }
    fclose(fp_result);

    // Clear the resource
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(org_arr_mem);
    ret = clReleaseMemObject(new_arr_mem);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(org_data_list);
    free(new_data_list);
    printf("\n\n%lu us\n", t2.tv_usec - t1.tv_usec);
    return 0;
}
