/* examples/hydro2d/pcl-hydro-params.cpp : parameter parsing for hydro

   (C) Romain Teyssier : CEA/IRFU           -- original F90 code
   (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
   (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
   (C) Jason Sewall : Intel -- 'pcl-hydro' optimized for modern x86
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
#include "pcl-hydro.hpp"

#include <cstring>
#include <cstdio>

static void default_values(hydro *H)
{
    // Default values should be given
    H->global_n[0]    = 20;
    H->global_n[1]    = 20;
    H->nxystep        = -1;
    H->dx             = 1.0;
    H->t              = 0.0;
    H->step           = 0;
    H->tend           = 0.0;
    H->courant_number = 0.5;
    H->iorder         = 2;
    H->slope_type     = 1.;
    H->scheme         = hydro::MUSCL;
    H->nstepmax       = (unsigned int)-1;
    H->testcase       = 0;
}

static void keyval(char *buffer, char **pkey, char **pval)
{
    char *ptr;
    *pkey = buffer;
    *pval = buffer;

    // kill the newline
    *pval = strchr(buffer, '\n');
    if (*pval)
        **pval = 0;

    // suppress leading whites or tabs
    while ((**pkey == ' ') || (**pkey == '\t'))
        (*pkey)++;
    *pval = strchr(buffer, '=');
    if (*pval) {
        **pval = 0;
        (*pval)++;
    }
    // strip key from white or tab
    while ((ptr = strchr(*pkey, ' ')) != NULL) {
        *ptr = 0;
    }
    while ((ptr = strchr(*pkey, '\t')) != NULL) {
        *ptr = 0;
    }
}

bool hydro_set_kv(hydro *H, char *kvstr)
{
    char *pkey, *pval;
    keyval(kvstr, &pkey, &pval);

    if(!pkey || !pval)
        return false;

    // int parameters
    if (strcmp(pkey, "nstepmax") == 0) {
        sscanf(pval, "%u", &H->nstepmax);
        return true;
    }
    if (strcmp(pkey, "nx") == 0) {
        int tmp;
        sscanf(pval, "%d", &tmp);
        if(tmp > 0)
        {
            H->global_n[0] = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    if (strcmp(pkey, "ny") == 0) {
        int tmp;
        sscanf(pval, "%d", &tmp);
        if(tmp > 0)
        {
            H->global_n[1] = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    if (strcmp(pkey, "nxystep") == 0) {
        int tmp;
        sscanf(pval, "%d", &tmp);
        if(tmp > 0)
        {
            H->nxystep = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    if (strcmp(pkey, "iorder") == 0) {
        int tmp;
        sscanf(pval, "%d", &tmp);
        if(tmp == 1 || tmp == 2)
        {
            H->iorder = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    // float parameters
    if (strcmp(pkey, "slope_type") == 0) {
        double tmp;
        sscanf(pval, REAL_FMT, &tmp);
        if(tmp > 0.0)
        {
            H->slope_type = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    if (strcmp(pkey, "tend") == 0) {
        double tmp;
        sscanf(pval, REAL_FMT, &tmp);
        if(tmp > 0.0)
        {
            H->tend = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    if (strcmp(pkey, "dx") == 0) {
        double tmp;
        sscanf(pval, REAL_FMT, &tmp);
        if(tmp > 0.0)
        {
            H->dx = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    if (strcmp(pkey, "courant_factor") == 0) {
        double tmp;
        sscanf(pval, REAL_FMT, &tmp);
        if(tmp > 0.0)
        {
            H->courant_number = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    if (strcmp(pkey, "testcase") == 0) {
        int tmp;
        sscanf(pval, "%d", &tmp);
        if(tmp == 0 || tmp == 1 || tmp == 2)
        {
            H->testcase = tmp;
            return true;
        }
        else
        {
            return false;
        }
    }
    // string parameter
    if (strcmp(pkey, "scheme") == 0) {
        if (strcmp(pval, "muscl") == 0) {
            H->scheme = hydro::MUSCL;
        } else if (strcmp(pval, "plmde") == 0) {
            H->scheme = hydro::PLMDE;
        } else if (strcmp(pval, "collela") == 0) {
            H->scheme = hydro::COLLELA;
        } else {
            return false;
        }
        return true;
    }
    return false;
}

static void process_input(hydro *H, const char *datafile, int quiet)
{
    FILE *fd = NULL;
    char buffer[1024];

    fd = xfopen_read(datafile, "r");
    if (fd == NULL) {
        fprintf(stderr, "can't read input file\n");
        exit(1);
    }
    while (fgets(buffer, 1024, fd) == buffer) {
        bool res = hydro_set_kv(H, buffer);
        if(!res && quiet < 2)
            printf("[PARAMS] Skipping unused key %s\n", buffer);
    }
    fclose(fd);
}

bool load_hydro_params(hydro *h, const char *file, int quiet)
{
    default_values(h);
    if(file)
        process_input(h, file, quiet);
    return true;
}
