///     Header file for calcIR.cu program       ///

#ifndef CALCIR_H 
#define CALCIR_H

// HEADERS

#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xdrfile/xdrfile.h>
#include <xdrfile/xdrfile_xtc.h>
#include <fftw3.h>


#ifdef USE_DOUBLES

typedef double  user_real_t;
typedef double complex user_complex_t;

#else

typedef float user_real_t;
typedef float complex user_complex_t;

#endif

// CONSTANTS

#define HBAR        5.308837367       // in cm-1 * ps
#define PI          3.14159265359
#define MAX_STR_LEN 80
#define PSTR        "||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PWID        50
#define CP_WRITE    0
#define CP_READ     1
#define CP_INIT     3
#define MALLOC_ERR  { printf(">>> ERROR on CPU: out of memory.\n"); exit(EXIT_FAILURE);}


// FUNCTIONS

void get_eproj( rvec *x, float boxx, float boxy, float boxz, int natoms, int natom_mol, 
                int nchrom, int nchrom_mol, int nmol, int model, user_real_t *eproj, 
                int *nlisti, int *nlistj, long long int *nlistFill, long long int nlistFillMax,
                user_real_t cplCut );


void get_kappa_sparse( rvec *x, float boxx, float boxy, float boxz, int natoms, int natom_mol, 
                       int nchrom, int nchrom_mol, int nmol, user_real_t *eproj, user_real_t *kappa, 
                       user_complex_t *mux, user_complex_t *muy, user_complex_t *muz, user_real_t avef, 
                       long long int *kappaFill, long long int kappaFillMax, int *kappai, int *kappaj,
                       int *nlisti, int *nlistj, long long int nlistFill );

user_real_t minImage( user_real_t dx, user_real_t boxl );

user_real_t mag3( user_real_t dx[3] );

user_real_t dot3( user_real_t x[3], user_real_t y[3] );


void ir_init( char *argv[], char gmxf[], char cptf[], char outf[], char model[], 
              user_real_t *dt, int *ntcfpoints,  int *nsamples, 
              int *sampleEvery, user_real_t *t1, user_real_t *avef,
              int *natom_mol, int *nchrom_mol, int *nzeros, user_real_t *beginTime,
              user_real_t *cplCut, int *maxCouple );

void printProgress( int currentStep, int totalSteps );

void checkpoint( char *argv[], char gmxf[], char cptf[], char outf[], char model[], 
                 user_real_t *dt, int *ntcfpoints, int *nsamples, int *sampleEvery, 
                 user_real_t *t1, user_real_t *avef, int *natom_mol, int *nchrom_mol, 
                 int *nzeros, user_real_t *beginTime, int nchrom, int *currentSample, 
                 int *currentFrame, user_complex_t *tcf, user_complex_t *cmux0, 
                 user_complex_t *cmuy0, user_complex_t *cmuz0, user_real_t *cplCut,
                 int *maxCouple, int RWI_FLAG );

void signal_handler( int sig );
#endif
