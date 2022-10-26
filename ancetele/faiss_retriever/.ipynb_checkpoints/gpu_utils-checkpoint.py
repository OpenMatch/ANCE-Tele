import faiss
import numpy as np
import pickle
import os
import glob


def clean_faiss_gpu():
    ngpu = faiss.get_num_gpus()
    tempmem = 0
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
            
            
            
def get_gpu_index(cpu_index):
    gpu_resources = []
    ngpu = faiss.get_num_gpus()
    tempmem = -1
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    def make_vres_vdev(i0=0, i1=-1):
        " return vectors of device ids and resources useful for gpu_multiple"
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        if i1 == -1:
            i1 = ngpu
        for i in range(i0, i1):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        return vres, vdev

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True 
    gpu_vector_resources, gpu_devices_vector = make_vres_vdev(0, ngpu)
    gpu_index = faiss.index_cpu_to_gpu_multiple(gpu_vector_resources, gpu_devices_vector, cpu_index, co)
    return gpu_index