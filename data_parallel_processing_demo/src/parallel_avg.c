#pragma warning (disable: 4996)

#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <stdio.h>
#include <CL/cl.h>
#include <time.h>
#include <sys/time.h>

int main()
{
    // Create the data list
    const unsigned int rowNum = 10000;
    const unsigned int colNum = 10000;
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

    struct timeval t1, t2;

    // Load the kernel codes into source_str
    FILE *fp_kernel = NULL;
    char *source_str = NULL;
    size_t source_size = 0;

    fp_kernel = fopen("src/parallel_avg_cl_kernel.cl", "r");
    if(!fp_kernel)
    {
        printf("Fail to open the kernel file!\n");
        free(data_arr);
        return -1;
    }
    
    fseek(fp_kernel, 0, SEEK_END);
    source_size = ftell(fp_kernel);
    fseek(fp_kernel, 0, SEEK_SET);
    if(!source_size)
    {
        printf("Empty kernel file!\n");
        fclose(fp_kernel);
        free(data_arr);
        return -1;
    }
    source_str = (char*)malloc(source_size);
    if(!source_str)
    {
        printf("Unable to assign the space for the kernel file codes!\n");
        fclose(fp_kernel);
        free(data_arr);
        return -1;
    }
    memset(source_str, 0, source_size);
    fread(source_str, 1, source_size, fp_kernel);
    fclose(fp_kernel);

    // Obtain the platform and devices information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platform;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platform);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create context for opencl
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    // Create command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    // Create buffer for each vector on the device
    unsigned int tbl_parameter[2] = {rowNum, colNum};
    cl_mem original_data_tbl_memobj = clCreateBuffer(context, CL_MEM_READ_ONLY, data_arr_size, NULL, &ret);
    cl_mem processing_data_tbl_memobj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_arr_size, NULL, &ret);
    cl_mem data_tbl_parameter = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(tbl_parameter), NULL, &ret);

    // Copy the data_arr into the GPU memory 
    ret = clEnqueueWriteBuffer(command_queue, original_data_tbl_memobj, CL_TRUE, 0, data_arr_size, data_arr, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, data_tbl_parameter, CL_TRUE, 0, sizeof(tbl_parameter), tbl_parameter, 0, NULL, NULL);

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    // Construct program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    // Create opencl kernel
    cl_kernel kernel = clCreateKernel(program, "avg_processing", &ret);

    // Set kernel parameters
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&original_data_tbl_memobj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&processing_data_tbl_memobj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&data_tbl_parameter);

    // Execute the kernel codes
    size_t global_item_num = data_arr_num;      // Handle the whole list
    size_t local_item_num = 64;                 // Cut into 64 pieces
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_num, &local_item_num, 0, NULL, NULL);

    // Read GPU buffer into the local ram buffer
    float *result_arr = (float*)malloc(data_arr_size);
    if(!result_arr)
    {
        printf("Unable to assign the space for result table!\n");
        free(data_arr);
        free(source_str);
        return -1;
    }
    gettimeofday(&t1, NULL);
    ret = clEnqueueReadBuffer(command_queue, processing_data_tbl_memobj, CL_TRUE, 0, data_arr_size, result_arr, 0, NULL, NULL);
    gettimeofday(&t2, NULL);

    // Clear the resource
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(original_data_tbl_memobj);
    ret = clReleaseMemObject(processing_data_tbl_memobj);
    ret = clReleaseMemObject(data_tbl_parameter);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(data_arr);
    free(result_arr);

    printf("\nParallel data processing done.\n%u ms elapsed.\n\n", ((unsigned int)t2.tv_usec - (unsigned int)t1.tv_usec) / 1000);

    return 0;
}
