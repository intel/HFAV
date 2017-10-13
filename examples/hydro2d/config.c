/* examples/hydro2d/config.c: print out configuration of software and hardware

   (C) Jason Sewall : Intel -- inital version
   (C) John Pennycook : Intel -- augmentations to above version
*/
/*
  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.
*/

#define USE_OMP
#define USE_NUMACTL
#include <gnu/libc-version.h>
#include <unistd.h>
#include <error.h>
#include <string.h>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_MKL
#include "mkl.h"
#endif
#ifdef USE_OMP
#include <omp.h>
#endif
#ifdef USE_NUMACTL
#include <numa.h>
#endif
#include <sys/utsname.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdarg>

static char config_null[] = "<undefined>";

static const char *xgetenv_name(const char *str)
{
    const char *res = getenv(str);
    if(res == 0)
        return config_null;
    else
        return res;
}

#ifdef USE_NUMACTL
struct node
{
    struct bitmask *cpus;
    int has_cpu;
    int nearest_memonly;
};

static void readnodes(struct node *nodes, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        nodes[i].cpus = numa_allocate_cpumask();
        if (nodes[i].cpus == 0)
            perror("allocate cpu bitmask");
        int ret = numa_node_to_cpus(i, nodes[i].cpus);
        if (ret != 0)
            perror("numa_node_to_cpus");
        nodes[i].has_cpu = numa_bitmask_weight(nodes[i].cpus);
        nodes[i].nearest_memonly = -1;
    }
}

static void findmem(struct node *nodes, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        if (nodes[i].has_cpu == 0)
            continue;
        // look for a memory-only node with closest distance
        int memidx = -1;
        int distance = 0x7FFFFFFF;
        int j;
        for (j = 0; j < n; ++j)
        {
            if (nodes[j].has_cpu != 0)
                continue;
            int d = numa_distance(i, j);
            if (d < distance)
            {
                distance = d;
                memidx = j;
            }
        }
        nodes[i].nearest_memonly = memidx;
    }
}

static int xsnprintf(char *str, size_t n, const char *fmt, ...)
{
    va_list val;
    va_start(val, fmt);
    int wanted_out = vsnprintf(str, n, fmt, val);
    va_end(val);
    if(wanted_out > n)
    {
        fprintf(stderr, "Ran out of buffer space for output string!\n");
        exit(1);
    }
    return wanted_out;
}

static char *mem_nodes(struct node *nodes, struct bitmask *mynodes, int nnodes)
{
    int i;
    char  temp[1024];
    memset(temp, 0, sizeof(char)*1024);
    char *curr = temp;
    *curr = 0;
    for (i = 0; i < nnodes; ++i)
    {
        if (numa_bitmask_isbitset(mynodes, i) &&
            nodes[i].nearest_memonly > 0)
        {
            if (curr != temp)
            {
                curr += snprintf(curr, 1023-(curr-temp), ",");
            }
            curr += snprintf(curr, 1023-(curr-temp), "%d", nodes[i].nearest_memonly);
        }
    }
    char *res = strdup(temp);
    if(!res)
        return config_null;
    return res;
}

static char *cpu_nodes(struct bitmask *mynodes, int nnodes)
{
    int i;
    char  temp[1024];
    memset(temp, 0, sizeof(char)*1024);
    char *curr = temp;
    for (i = 0; i < nnodes; ++i)
    {
        if (numa_bitmask_isbitset(mynodes, i))
        {
            if (curr != temp)
            {
                curr += snprintf(curr, 1023-(curr-temp), ",");
            }
            curr += snprintf(curr, 1023-(curr-temp), "%d", i);
        }
    }
    char *res = strdup(temp);
    if(!res)
        return config_null;
    return res;
}
#endif

static char *format_uname()
{
    struct utsname un;
    if(uname(&un) == -1)
    {
        perror("uname");
        exit(1);
    }
    char buff[1024];
    snprintf(buff, 1023, "%s-%s-%s-%s", un.nodename, un.machine, un.sysname, un.release);
    char *res = strdup(buff);
    if(!res)
        return config_null;
    return res;
}

static void cpuid(const unsigned int info, unsigned int *eax, unsigned int *ebx, unsigned int *ecx, unsigned int *edx)
{
    __asm__("cpuid;"
            :"=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx)
            :"a" (info));
}

static void vendorid(char id[13])
{
    unsigned int temp;
    cpuid(0, &temp, (unsigned int*)id, (unsigned int*)(id+8), (unsigned int*)(id+4));
    id[12] = 0;
}

static void proc_brand(char str[49])
{
    unsigned int i;
    static char nope[] = "Unknown";
    unsigned int okay;
    cpuid(0x80000000, &okay, (unsigned int*)str, (unsigned int*)(str+4), (unsigned int*)(str+8));
    if(okay < 0x80000004)
    {
        strcpy(str, nope);
    }
    else
    {
        for(i = 0; i < 3; ++i)
        {
            cpuid(0x80000002+i, (unsigned int*)(str+16*i), (unsigned int*)(str+16*i+4), (unsigned int*)(str+16*i+8), (unsigned int*)(str+16*i+12));
        }
    }
    str[48] = 0;
}

typedef struct cpuinfo
{
    unsigned stepping : 4;
    unsigned model : 4;
    unsigned family_id : 4;
    unsigned proc_type : 2;
    unsigned nothing : 2;
    unsigned extended_model_id : 4;
    unsigned extended_family_id : 8;
    unsigned nothing2 : 6;
    unsigned display_family;
    unsigned display_model;
} cpuinfo;

static void cpu_info(struct cpuinfo *ci)
{
    unsigned int b, c, d;
    cpuid(0x1, (unsigned int*)ci, &b, &c, &d);
    if(ci->family_id == 0x0F)
    {
        ci->display_family = ci->extended_family_id + ci->family_id;
    }
    else
    {
        ci->display_family = ci->family_id;
    }
    if(ci->family_id == 0x0F || ci->family_id == 0x06)
    {
        ci->display_model = (ci->extended_model_id << 4) + ci->model;
    }
    else
    {
        ci->display_model = ci->model;
    }
}

void print_config(FILE *fp)
{
    fprintf(fp, "%20s = %s\n", "GIT_VERSION", GIT_VERSION);
    fprintf(fp, "%20s = %s\n", "BUILD_HOST", BUILD_HOST);
    fprintf(fp, "%20s = %s\n", "COMPILER_VERSION", COMPILER_VERSION);
    fprintf(fp, "%20s = %s\n", "GLIBC_VERSION", gnu_get_libc_version ());
#ifdef USE_MKL
    char mkl_version[1024];
    mkl_get_version_string(mkl_version, 1024);
    fprintf(fp, "%20s = %s\n", "MKL_VERSION", mkl_version);
#endif
#ifdef USE_MPI
    int mpi_major,  mpi_minor;
    MPI_Get_version(&mpi_major, &mpi_minor);
    fprintf(fp, "%20s = %d.%d\n", "MPI_VERSION", mpi_major, mpi_minor);
    int mpi_len;
    char mpi_library_version[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Get_library_version(mpi_library_version, &mpi_len);
    char *mpi_newline = strchr(mpi_library_version, '\n');
    if(mpi_newline)
        *mpi_newline = 0;
    fprintf(fp, "%20s = %s\n", "MPI_LIBRARY_VERSION", mpi_library_version);
#endif
    fprintf(fp, "%20s = %s %s\n", "BUILD_DATE", __DATE__, __TIME__);
    fprintf(fp, "\n");
    char *host = format_uname();
    fprintf(fp, "%20s = %s\n", "HOST", host);
    if(host != config_null)
        free(host);
    char vid[13];
    vendorid(vid);
    char pb[49];
    proc_brand(pb);
    fprintf(fp, "%20s = %s %s\n", "CPU", vid, pb);
    cpuinfo ci;
    cpu_info(&ci);
    fprintf(fp, "%20s = %s\n", "LD_PRELOAD", xgetenv_name("LD_PRELOAD"));
    fprintf(fp, "%20s = Family %u Model %u Stepping %u\n", "CPUINFO", ci.display_family, ci.display_model, ci.stepping);
    fprintf(fp, "\n");
    #ifdef USE_OMP
    fprintf(fp, "%20s = %d\n", "NTHREADS", omp_get_max_threads());
    fprintf(fp, "%20s = %s\n", "KMP_AFFINITY", xgetenv_name("KMP_AFFINITY"));
    fprintf(fp, "%20s = %s\n", "KMP_PLACE_THREADS", xgetenv_name("KMP_PLACE_THREADS"));
    fprintf(fp, "%20s = %s\n", "KMP_BLOCKTIME", xgetenv_name("KMP_BLOCKTIME"));
    fprintf(fp, "\n");
#endif
#ifdef USE_NUMACTL
    if(numa_available() != -1)
    {
        fprintf(fp, "%20s = %s\n", "NUMA_AVAILABLE", "YES");
        const int nnodes = numa_max_node()+1;
        fprintf(fp, "%20s = %d\n", "NUMA_NODES", nnodes);
        struct node nodes[nnodes];
        readnodes(nodes, nnodes);
        findmem(nodes, nnodes);
        struct bitmask *mynodes = numa_allocate_nodemask();
        struct bitmask *mycpus = numa_allocate_cpumask();
        int ret = numa_sched_getaffinity(0, mycpus);
        if (ret <= 0)       // returns # bytes copied
            perror("numa_sched_getaffinity");
        const int ncpus = numa_num_possible_cpus();
        int i;
        for (i = 0; i < nnodes; ++i)
        {
            // check if there is any intersection with this node
            int j;
            for (j = 0; j < ncpus; ++j)
                if (numa_bitmask_isbitset(mycpus, j) &&
                    numa_bitmask_isbitset(nodes[i].cpus, j))
                    numa_bitmask_setbit(mynodes, i);
        }
        numa_free_cpumask(mycpus);
        char *mem_node_str = mem_nodes(nodes, mynodes, nnodes);
        fprintf(fp, "%20s = %s\n", "NUMA_MEM_NODES", mem_node_str);
        if(mem_node_str != config_null)
            free(mem_node_str);
        char *cpu_node_str = cpu_nodes(mynodes, nnodes);
        fprintf(fp, "%20s = %s\n", "NUMA_CPU_NODES", cpu_node_str);
        if(cpu_node_str != config_null)
            free(cpu_node_str);
    }
    else
    {
        fprintf(fp, "%20s = %s\n", "NUMA_AVAILABLE", "NO");
    }
    fprintf(fp, "\n");
#endif
    fprintf(fp, "\n");
}

#ifdef TEST_PROG
int main()
{
    print_config(stderr);
    return 0;
}
#endif
