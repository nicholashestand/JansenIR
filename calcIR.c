/*  This program calculates the OH stetch IR absorption spectrum
 *  for coupled water from an MD trajectory. The exciton Hamilt-
 *  onian is built using the maps developed by Skinner  and  co-
 *  workers. In this version, I use Jansens approximation to pr-
 *  opigate the system hamiltonian. It is especially useful  for
 *  very large systems. For systems with less than 1000 molecul-
 *  es, the GPU version should work sufficiently quickly.
 */


#include "calcIR.h" 


// TODO: Get rid of xyz and just make mu a scalar


int main(int argc, char *argv[])
{

    // start mpi
    int ierr, nproc, rank, istart, iend;
    ierr = MPI_Init(&argc, &argv);
    if ( ierr != MPI_SUCCESS )
    {
        printf("Error starting MPI. Aborting...\n");
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }

    MPI_Comm_size( MPI_COMM_WORLD, &nproc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // Some help for starting the program. User must supply a single argument
    if ( argc != 2 ){
        mpiprintf("Usage:\n"
                  "\tInclude as the first argument the name of an input file. No other artuments are \n"
                  "\tallowed.\n");
        MPI_Finalize();
        exit(EXIT_SUCCESS);
    }

    

    // ***              Variable Declaration            *** //
    // **************************************************** //

    mpiprintf("\n>>> Setting default parameters\n");

    // Model parameters
    char          gmxf[MAX_STR_LEN]; strncpy( gmxf, " ", MAX_STR_LEN );   // trajectory file
    char          outf[MAX_STR_LEN]; strncpy( outf, " ", MAX_STR_LEN );   // name for output files
    char          cptf[MAX_STR_LEN]; strncpy( cptf, " ", MAX_STR_LEN );   // name for output files
    char          model[MAX_STR_LEN];strncpy( model," ", MAX_STR_LEN );   // water model tip4p, tip4p2005, e3b2, e3b3
    int           imodel        = 0;                                      // integer for water model
    int           ntcfpoints    = 150 ;                                   // the number of tcf points for each spectrum
    int           nsamples      = 1   ;                                   // number of samples to average for the total spectrum
    int           sampleEvery   = 10  ;                                   // sample a new configuration every sampleEvery ps. Note the way the program is written, 
                                                                          // ntcfpoints*dt must be less than sampleEvery.
    int           natom_mol     = 4;                                      // Atoms per water molecule  :: MODEL DEPENDENT
    int           nchrom_mol    = 2;                                      // Chromophores per molecule :: TWO for stretch -- ONE for bend
    int           nzeros        = 25600;                                  // zeros for padding fft

    user_real_t   dt            = 0.010;                                  // dt between frames in xtc file (in ps)
    user_real_t   beginTime     = 0    ;                                  // the beginning time in ps to allow for equilibration, if desired
    user_real_t   t1            = 0.260;                                  // relaxation time ( in ps )
    user_real_t   avef          = 3415.2;                                 // the approximate average stretch frequency to get rid of high
                                                                          // frequency oscillations in the time correlation function
    user_real_t   cplCut        = 1.0;                                    // O-O cutoff for coupling calculations in nm
    int           maxCouple     = 100;                                    // maximum number of coupled molecules


    // read in model parameters
    // START FROM INPUT FILE
    ir_init( argv, gmxf, cptf, outf, model, &dt, &ntcfpoints, &nsamples, &sampleEvery, &t1, 
            &avef, &natom_mol, &nchrom_mol, &nzeros, &beginTime, &cplCut, &maxCouple, rank );


    // Print the parameters to stdout
    mpiprintf("\tSetting xtc file %s\n",                       gmxf        );
    mpiprintf("\tSetting output file name to %s\n",            outf        );
    mpiprintf("\tSetting cpt file %s\n",                       cptf        );
    mpiprintf("\tSetting model to %s\n",                       model       );
    mpiprintf("\tSetting the number of tcf points to %d\n",    ntcfpoints  );
    mpiprintf("\tSetting nsamples to %d\n",                    nsamples    ); 
    mpiprintf("\tSetting sampleEvery to %d (ps)\n",            sampleEvery );
    mpiprintf("\tSetting natom_mol to %d\n",                   natom_mol   );
    mpiprintf("\tSetting nchrom_mol to %d\n",                  nchrom_mol  );
    mpiprintf("\tSetting nzeros to %d\n",                      nzeros      );
    mpiprintf("\tSetting maxCouple to %d\n",                   maxCouple   );
#ifdef USE_DOUBLES
    mpiprintf("\tSetting dt to %lf\n",                         dt          );
    mpiprintf("\tSetting t1 to %lf (ps)\n",                    t1          );
    mpiprintf("\tSetting avef to %lf\n",                       avef        );
    mpiprintf("\tSetting equilibration time to %lf (ps)\n",    beginTime   );
    mpiprintf("\tSetting cplCut to %lf\n",                     cplCut      );
#else
    mpiprintf("\tSetting dt to %f\n",                          dt          );
    mpiprintf("\tSetting t1 to %f (ps)\n",                     t1          );
    mpiprintf("\tSetting avef to %f\n",                        avef        );
    mpiprintf("\tSetting equilibration time to %f (ps)\n",     beginTime   );
    mpiprintf("\tSetting cplCut to %f\n",                      cplCut      );
#endif

    // set imodel based on model passed...if 1, reset OM lengths to tip4p lengths
    if ( strcmp( model, "tip4p2005" ) == 0 || strcmp( model, "e3b3" ) == 0 ) imodel = 1;
    else imodel = 0;
 
    // divide up samples for different mpi processes
    istart = rank*nsamples/nproc; iend = (rank+1)*nsamples/nproc,nsamples;

    // Useful variables and condstants
    int                 natoms, nmol, nchrom;                                           // number of atoms, molecules, chromophores
    int                 currentSample   = 0;                                            // current sample
    int                 currentFrame    = 0;                                            // current frame
    const int           ntcfpointsR     = ( nzeros + ntcfpoints - 1 ) * 2;              // number of points for the real fourier transform
    long long int       nchrom2;                                                        // nchrom squared
    long long int       kappaFill, kappaFillMax;                                        // number of nonzero hamiltonian elements
    long long int       nlistFill, nlistFillMax;                                        // number of neighbors in neighborlist
    int                 ii, jj;                                                         // indices for sparse matrix


    // Trajectory variables for the CPU
    rvec                *x;                                                             // Position vector
    matrix              box;                                                            // Box vectors
    float               gmxtime, prec;                                                  // Time at current frame, precision of xtf file
    int                 step, xdrinfo;                                                  // The current step number


    // Spectroscopy Variables
    user_complex_t      *cmux0, *cmuy0, *cmuz0;                                         // complex version of the transition dipole moment at t=0 
    user_complex_t      *cmux,  *cmuy,  *cmuz;                                          // complex versions of the transition dipole moment
    user_complex_t      *tmpmu;                                                         // to sum all polarizations
    user_real_t         *eproj;                                                         // the electric field projected along the oh bonds
    user_real_t         *kappa;                                                         // the hamiltonian
    int                 *kappai, *kappaj;                                               // to keep the hamiltonian sparse
    int                 *nlisti, *nlistj;                                               // to keep neighborlist indices

    // For Jansen propigation
    user_complex_t      *A0_2;                                                          // to propigate mu0
    user_complex_t      cmutmp0, cmutmp1;                                               // temporary for matrix multiplications
    user_real_t         Jij, co, si;                                                    // for Aijk
    
    // variables for spectrum calculations
    user_real_t         *w;                                                             // Eigenvalues on the GPU
    user_real_t         *omega;                                                         // Frequencies on CPU and GPU
    user_real_t         *Sw;                                                            // Spectral density on CPU and GPU
    user_real_t         *tmpSw;                                                         // Temporary spectral density

    // variables for TCF
    user_complex_t      tcfx, tcfy, tcfz;                                               // Time correlation function, polarized
    user_complex_t      dcy, tcftmp;                                                    // Decay constant and a temporary variable for the tcf
    user_complex_t      *pdtcf;                                                         // padded time correlation functions
    user_complex_t      *tcf;                                                           // Time correlation function
    user_real_t         *Ftcf;                                                          // Fourier transformed time correlation function

    // fftw
    fftw_plan plan;


    // for timing and errors
    time_t              start=time(NULL), end;

    // for file output
    FILE *rtcf;
    FILE *itcf;
    FILE *spec_density;
    FILE *spec_lineshape; 
    char *fname;
    fname = (char *) malloc( strlen(outf) + 9 );
    user_real_t factor;                                                                 // conversion factor to give energy and correct intensity from FFT
    user_real_t freq;
    

    // **************************************************** //
    // ***         End  Variable Declaration            *** //


    



    // ***          Begin main routine                  *** //
    // **************************************************** //


    // Open trajectory file and get info about the system
    XDRFILE *trj = xdrfile_open( gmxf, "r" ); 
    if ( trj == NULL )
    {
        mpiprintf("WARNING: The file %s could not be opened. Is the name correct?\n", gmxf);
        exit(EXIT_FAILURE);
    }

    read_xtc_natoms( (char *)gmxf, &natoms);
    nmol         = natoms / natom_mol;
    nchrom       = nmol * nchrom_mol;
    nchrom2      = (long long int) nchrom*nchrom;
    kappaFillMax = (long long int) nchrom*maxCouple*2 + nchrom;
    nlistFillMax = (long long int) nchrom*maxCouple   + nchrom;

    mpiprintf(">>> Will read the trajectory from: %s.\n",gmxf);
    mpiprintf(">>> Found %d atoms and %d molecules.\n",natoms, nmol);
    mpiprintf(">>> Found %d chromophores.\n",nchrom);


    // ***              MEMORY ALLOCATION               *** //
    // **************************************************** //


    // CPU arrays
    x       = (rvec*)            malloc( natoms       * sizeof(x[0] ));             if ( x == NULL )    MALLOC_ERR;
    tcf     = (user_complex_t *) calloc( ntcfpoints   , sizeof(user_complex_t));    if ( tcf == NULL )  MALLOC_ERR;
    Ftcf    = (user_real_t *)    calloc( ntcfpointsR  , sizeof(user_real_t));       if ( Ftcf == NULL ) MALLOC_ERR;
    eproj   = (user_real_t *)    calloc( nchrom       , sizeof(user_real_t));       if ( eproj == NULL) MALLOC_ERR;
    cmux    = (user_complex_t *) calloc( nchrom       , sizeof(user_complex_t));    if ( cmux  == NULL) MALLOC_ERR;
    cmuy    = (user_complex_t *) calloc( nchrom       , sizeof(user_complex_t));    if ( cmuy  == NULL) MALLOC_ERR;
    cmuz    = (user_complex_t *) calloc( nchrom       , sizeof(user_complex_t));    if ( cmuz  == NULL) MALLOC_ERR;
    cmux0   = (user_complex_t *) calloc( nchrom       , sizeof(user_complex_t));    if ( cmux0 == NULL) MALLOC_ERR;
    cmuy0   = (user_complex_t *) calloc( nchrom       , sizeof(user_complex_t));    if ( cmuy0 == NULL) MALLOC_ERR;
    cmuz0   = (user_complex_t *) calloc( nchrom       , sizeof(user_complex_t));    if ( cmuz0 == NULL) MALLOC_ERR;
    tmpmu   = (user_complex_t *) calloc( nchrom       , sizeof(user_complex_t));    if ( tmpmu == NULL) MALLOC_ERR;
    kappa   = (user_real_t *)    calloc( kappaFillMax , sizeof(user_real_t));       if ( kappa == NULL) MALLOC_ERR;
    kappai  = (int *)            calloc( kappaFillMax , sizeof(int));               if ( kappai== NULL) MALLOC_ERR;
    kappaj  = (int *)            calloc( kappaFillMax , sizeof(int));               if ( kappaj== NULL) MALLOC_ERR;
    nlisti  = (int *)            calloc( nlistFillMax , sizeof(int));               if ( nlisti== NULL) MALLOC_ERR;
    nlistj  = (int *)            calloc( nlistFillMax , sizeof(int));               if ( nlistj== NULL) MALLOC_ERR;
    A0_2    = (user_complex_t *) calloc( nchrom       , sizeof(user_complex_t));    if ( A0_2  == NULL) MALLOC_ERR;

    // ***            END MEMORY ALLOCATION             *** //
    // **************************************************** //
    

    mpiprintf("\n>>> Now calculating the absorption spectrum\n");
    mpiprintf("----------------------------------------------------------\n");



    // **************************************************** //
    // ***          OUTER LOOP OVER SAMPLES             *** //

    currentSample = istart;

    while( currentSample < iend )
    {

        // search trajectory for current sample starting point
        xdrinfo = read_xtc( trj, natoms, &step, &gmxtime, box, x, &prec );
        if ( xdrinfo != 0 )
        {
            mpiprintf("WARNING:: read_xtc returned error %d.\n"
                   "Is the trajectory long enough?\n", xdrinfo);
            exit(EXIT_FAILURE);
        }

        if ( currentSample * sampleEvery + (int) beginTime == (int) gmxtime )
        {
            mpiprintf("\n    Now processing sample %d/%d starting at %.2f ps\n", currentSample + 1, iend, gmxtime );
            fflush(stdout);


            // **************************************************** //
            // ***         MAIN LOOP OVER TRAJECTORY            *** //
            while( currentFrame < ntcfpoints )
            {


                // ---------------------------------------------------- //
                // ***          Get Info About The System           *** //


                // read the current frame from the trajectory file
                // note it was read in the outer loop if we are at frame 0
                if ( currentFrame != 0 ) read_xtc( trj, natoms, &step, &gmxtime, box, x, &prec );

                // calculate the electric field projection along OH bonds and build the exciton hamiltonian
                get_eproj        ( x, box[0][0], box[1][1], box[2][2], natoms, natom_mol, nchrom, nchrom_mol, nmol, 
                                   imodel, eproj, nlisti, nlistj, &nlistFill, nlistFillMax, cplCut );
                get_kappa_sparse ( x, box[0][0], box[1][1], box[2][2], natoms, natom_mol, nchrom, nchrom_mol, nmol, 
                                   eproj, kappa, cmux, cmuy, cmuz, avef, &kappaFill, kappaFillMax, kappai, kappaj,
                                   nlisti, nlistj, nlistFill );

                // ***          Done getting System Info            *** //
                // ---------------------------------------------------- //


                // USING JANSEN's METHOD
                // ---------------------------------------------------- //
                // ***           Time Correlation Function          *** //


                // set TDM at t=0, this will be propigated 
                if ( currentFrame == 0 )
                {
                    // initialize cmu0
                    for ( int i = 0; i < nchrom; i++ )
                    {
                        cmux0[i] = cmux[i];
                        cmuy0[i] = cmuy[i];
                        cmuz0[i] = cmuz[i];
                    }
                }
                else
                {
                    // propigate the transition dipole moment vector using Jansen's approximation
                    // first build exp(A0/2) diagonal matrix
                    for ( int i = 0; i< kappaFill; i++ )   // loop over every nonzero element of the hamiltonian
                    {
                        ii = kappai[i]; 
                        jj = kappaj[i];
                        if ( ii == jj ) {                  // if this is a diagonal element, plug it in to the relevant part of A0
                            A0_2[ ii ] = cexp(-0.5*I*dt/HBAR * kappa[i] );
                        }
                    }

                    // multiply mu0 by exp(A0/2)
                    for ( int i = 0; i < nchrom; i++ )
                    {
                        cmux0[i] = A0_2[i]*cmux0[i];
                        cmuy0[i] = A0_2[i]*cmuy0[i];
                        cmuz0[i] = A0_2[i]*cmuz0[i];
                    }

                    // multiply by mu0 exp(Aijk)
                    for ( int i = 0; i < kappaFill; i++ ) // loop over every nonzero element of the hamiltonian
                    {
                        ii  = kappai[i];
                        jj  = kappaj[i];

                        // diagonal elements are not a part of A1, they belong to A0
                        if ( ii == jj ) continue;

                        // build the elements of the 2x2 matrix 
                        Jij = dt/HBAR * kappa[i];
                        co  = cos(Jij); si = sin(Jij);

                        // perform the multiplications on the x component
                        cmutmp0   =     co * cmux0[ii] - I * si * cmux0[jj];
                        cmutmp1   = -1.*I * si * cmux0[ii] +     co * cmux0[jj];
                        cmux0[ii] = cmutmp0; 
                        cmux0[jj] = cmutmp1;

                        // perform the multiplications on the y component
                        cmutmp0   =     co * cmuy0[ii] - I * si * cmuy0[jj];
                        cmutmp1   = -1.*I * si * cmuy0[ii] +     co * cmuy0[jj];
                        cmuy0[ii] = cmutmp0; 
                        cmuy0[jj] = cmutmp1;

                        // perform the multiplications on the z component
                        cmutmp0   =     co * cmuz0[ii] - I * si * cmuz0[jj];
                        cmutmp1   = -1.*I * si * cmuz0[ii] +     co * cmuz0[jj];
                        cmuz0[ii] = cmutmp0; 
                        cmuz0[jj] = cmutmp1;
                    }

                    // multiply mu0 by exp(A0/2) again
                    for ( int i = 0; i < nchrom; i++ )
                    {
                        cmux0[i] = A0_2[i]*cmux0[i];
                        cmuy0[i] = A0_2[i]*cmuy0[i];
                        cmuz0[i] = A0_2[i]*cmuz0[i];
                    }
                }

                // calculate the TCF. cmux*cmux0
                tcfx = 0.;
                tcfy = 0.;
                tcfz = 0.;
                for ( int i = 0; i < nchrom; i++ )
                {
                    tcfx += cmux[i] * cmux0[i];
                    tcfy += cmuy[i] * cmuy0[i];
                    tcfz += cmuz[i] * cmuz0[i];
                }

                // accumulate the tcf over the samples
                tcf[ currentFrame ] = tcf[ currentFrame ] + tcfx + tcfy + tcfz;

                // ***        Done with Time Correlation            *** //
                // ---------------------------------------------------- //


                // update progress bar if simulation is big enough, otherwise it really isn't necessary
                if ( nchrom > 400 && rank == 0) printProgress( currentFrame, ntcfpoints-1 );
            
                // done with current frame, move to next
                currentFrame += 1;
            }

            // done with current sample, move to next, and reset currentFrame to 0
            currentSample +=1;
            currentFrame  = 0;

        }
    } // end outer loop

    // close xdr file
    xdrfile_close(trj);


    // reduce tcf to root -- MPI has not complex data type so I have to split this into reals for reduction
    user_real_t *rtcfa, *itcfa, *rtcfsum, *itcfsum;
    rtcfa   = ( user_real_t *) calloc( ntcfpoints, sizeof( user_real_t ) ); if ( rtcf    == NULL )    MALLOC_ERR;
    itcfa   = ( user_real_t *) calloc( ntcfpoints, sizeof( user_real_t ) ); if ( itcf    == NULL )    MALLOC_ERR;
    rtcfsum = ( user_real_t *) calloc( ntcfpoints, sizeof( user_real_t ) ); if ( rtcfsum == NULL )    MALLOC_ERR;
    itcfsum = ( user_real_t *) calloc( ntcfpoints, sizeof( user_real_t ) ); if ( itcfsum == NULL )    MALLOC_ERR;
 
    for ( int i = 0; i < ntcfpoints; i++ )
    {
        rtcfa[i] = creal(tcf[i]);
        itcfa[i] = cimag(tcf[i]);
    }

#ifdef USE_DOUBLES
    ierr = MPI_Reduce( rtcfa, rtcfsum, ntcfpoints, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
    ierr = MPI_Reduce( itcfa, itcfsum, ntcfpoints, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
#else
    ierr = MPI_Reduce( rtcfa, rtcfsum, ntcfpoints, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD );
    ierr = MPI_Reduce( itcfa, itcfsum, ntcfpoints, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD );
#endif
    mpiprintf("\n\n----------------------------------------------------------\n");
    mpiprintf("All samples now reduced to rank 0.\n Finishing up.");

    if ( rank == 0 )
    {
 
        for ( int i = 0; i < ntcfpoints; i++ )
        {
            tcf[i] = rtcfsum[i] + I*itcfsum[i];
        }

        pdtcf = (user_complex_t *) calloc( ntcfpoints+nzeros, sizeof(user_complex_t));
        // let root take the fft and write output files
        for ( int i = 0; i < ntcfpoints; i++ )
        {
            // multiply the tcf by the relaxation term
            dcy      = exp( -1.0 * i * dt / ( 2.0 * t1 ));
            tcf[i]   = tcf[i]*dcy;
            pdtcf[i] = tcf[i] / nsamples;
        }
        for ( int i = 0; i < nzeros; i++ ) pdtcf[i+ntcfpoints] = 0. + 0.*I;

        plan = fftw_plan_dft_c2r_1d( ntcfpoints + nzeros, pdtcf, Ftcf, FFTW_ESTIMATE );
        fftw_execute(plan);

        // write time correlation function
        rtcf = fopen(strcat(strcpy(fname,outf),"rtcf.dat"), "w");
        itcf = fopen(strcat(strcpy(fname,outf),"itcf.dat"), "w");
        for ( int i = 0; i < ntcfpoints; i++ )
        {
            fprintf( rtcf, "%g %g \n", i*dt, ( creal(tcf[i])) );
            fprintf( itcf, "%g %g \n", i*dt, ( cimag(tcf[i])) );
        }
        fclose( rtcf );
        fclose( itcf );

        // Write the absorption lineshape
        // Since the C2R transform is inverse by default, the frequencies have to be negated
        // NOTE: to compare with YICUN's code, divide Ftcf by 2
        spec_lineshape = fopen(strcat(strcpy(fname,outf),"spec.dat"),"w");
        factor         = 2*PI*HBAR/(dt*(ntcfpoints+nzeros));                // conversion factor to give energy and correct intensity from FFT
        for ( int i = (ntcfpoints+nzeros)/2; i < ntcfpoints+nzeros; i++ )   // "negative" FFT frequencies
        {
            freq = (i-ntcfpoints-nzeros)*factor + avef;
            fprintf(spec_lineshape, "%g %g\n", freq, Ftcf[i]/(factor*(ntcfpoints+nzeros)));
        }
        for ( int i = 0; i < ntcfpoints+nzeros / 2 ; i++)                   // "positive" FFT frequencies
        {
            freq = i*factor + avef;
            fprintf(spec_lineshape, "%g %g\n", freq, Ftcf[i]/(factor*(ntcfpoints+nzeros)));
        }
        fclose(spec_lineshape);
    }

    // free the variables
    free(x);
    free(tcf);
    free(Ftcf);
    free(eproj);
    free(cmux);
    free(cmuy);
    free(cmuz);
    free(cmux0);
    free(cmuy0);
    free(cmuz0);
    free(tmpmu);
    free(kappa);
    free(kappai);
    free(kappaj);
    free(nlisti);
    free(nlistj);
    free(A0_2);
    free(rtcfa);
    free(itcfa);
    free(rtcfsum);
    free(itcfsum);



    end = time(NULL);
    mpiprintf("\n>>> Done with the calculation in %f seconds.\n", difftime(end,start));

    MPI_Finalize();

    return 0;
}

/**********************************************************
   
   BUILD ELECTRIC FIELD PROJECTION ALONG OH BONDS
                    GPU FUNCTION

 **********************************************************/
void get_eproj( rvec *x, float boxx, float boxy, float boxz, int natoms, int natom_mol, 
                int nchrom, int nchrom_mol, int nmol, int model, user_real_t  *eproj,
                int *nlisti, int *nlistj, long long int *nlistFill, long long int nlistFillMax,
                user_real_t cplCut)
{
    
    int n, m, i, j, istart, istride;
    int chrom;
    user_real_t mox[DIM];                     // oxygen position on molecule m
    user_real_t mx[DIM];                      // atom position on molecule m
    user_real_t nhx[DIM];                     // hydrogen position on molecule n of the current chromophore
    user_real_t nox[DIM];                     // oxygen position on molecule n
    user_real_t nohx[DIM];                    // the unit vector pointing along the OH bond for the current chromophore
    user_real_t mom[DIM];                     // the OM vector on molecule m
    user_real_t dr[DIM];                      // the min image vector between two atoms
    user_real_t r;                            // the distance between two atoms 
    user_real_t dro;                          // distance between two oxygen atoms
    const float cutoff = 0.7831;              // the oh cutoff distance
    const float bohr_nm = 18.8973;            // convert from bohr to nanometer
    user_real_t efield[DIM];                  // the electric field vector

    // initialize nlistFill to zero
    *nlistFill = 0;

    // Loop over the chromophores belonging to the current thread
    for ( chrom = 0; chrom < nchrom; chrom ++ )
    {
        // calculate the molecule hosting the current chromophore 
        n = chrom / nchrom_mol;

        // initialize the electric field vector to zero at this chromophore
        efield[0]   =   0.;
        efield[1]   =   0.;
        efield[2]   =   0.;


        // ***          GET INFO ABOUT MOLECULE N HOSTING CHROMOPHORE       *** //
        //                      N IS OUR REFERENCE MOLECULE                     //
        // get the position of the hydrogen associated with the current stretch 
        // NOTE: I'm making some assumptions about the ordering of the positions, 
        // this can be changed if necessary for a more robust program
        // Throughout, I assume that the atoms are grouped into molecules and that
        // every 4th molecule starting at 0 (1, 2, 3) is OW (HW1, HW2, MW)
        if ( chrom % 2 == 0 ){      //HW1
            nhx[0]  = x[ n*natom_mol + 1 ][0];
            nhx[1]  = x[ n*natom_mol + 1 ][1];
            nhx[2]  = x[ n*natom_mol + 1 ][2];
        }
        else if ( chrom % 2 == 1 ){ //HW2
            nhx[0]  = x[ n*natom_mol + 2 ][0];
            nhx[1]  = x[ n*natom_mol + 2 ][1];
            nhx[2]  = x[ n*natom_mol + 2 ][2];
        }

        // The oxygen position
        nox[0]  = x[ n*natom_mol ][0];
        nox[1]  = x[ n*natom_mol ][1];
        nox[2]  = x[ n*natom_mol ][2];

        // The oh unit vector
        nohx[0] = minImage( nhx[0] - nox[0], boxx );
        nohx[1] = minImage( nhx[1] - nox[1], boxy );
        nohx[2] = minImage( nhx[2] - nox[2], boxz );
        r       = mag3(nohx);
        nohx[0] /= r;
        nohx[1] /= r;
        nohx[2] /= r;

        // add self to neighborlist
        if ( chrom % 2 == 0 ) // only do once
        {
            nlisti[*nlistFill] = n;
            nlistj[*nlistFill] = n;
            *nlistFill        += 1;
            if (*nlistFill >= nlistFillMax )
            {
                printf("nlistFill >= nlistFillMax (=%lld). Aborting!\n", nlistFillMax );
                exit(EXIT_FAILURE);
            }
        }

        // ***          DONE WITH MOLECULE N                                *** //



        // ***          LOOP OVER ALL OTHER MOLECULES                       *** //
        for ( m = 0; m < nmol; m++ ){

            // skip the reference molecule
            if ( m == n ) continue;

            // get oxygen position on current molecule
            mox[0] = x[ m*natom_mol ][0];
            mox[1] = x[ m*natom_mol ][1];
            mox[2] = x[ m*natom_mol ][2];

            // find o-o displacement and add to neighborlist if is short enough
            // I am doing this on a molecule basis, not a chromophore basis
            if ( chrom % 2 == 0 ) // only do once
            {
                dr[0]  = minImage( mox[0] - nox[0], boxx );
                dr[1]  = minImage( mox[1] - nox[1], boxy );
                dr[2]  = minImage( mox[2] - nox[2], boxz );
                dro    = mag3(dr);
                if ( dro < cplCut )
                {
                    nlisti[*nlistFill] = n;
                    nlistj[*nlistFill] = m;
                    *nlistFill        += 1;

                    if (*nlistFill >= nlistFillMax )
                    {
                        printf("nlistFill >= nlistFillMax (=%lld). Aborting!\n", nlistFillMax );
                        exit(EXIT_FAILURE);
                    }
                }
            }

            // find displacement between oxygen on m and hydrogen on n
            dr[0]  = minImage( mox[0] - nhx[0], boxx );
            dr[1]  = minImage( mox[1] - nhx[1], boxy );
            dr[2]  = minImage( mox[2] - nhx[2], boxz );
            r      = mag3(dr);

            // skip if the distance is greater than the cutoff
            if ( r > cutoff ) continue;

            // loop over all atoms in the current molecule and calculate the electric field 
            // (excluding the oxygen atoms since they have no charge)
            for ( i=1; i < natom_mol; i++ ){

                // position of current atom
                mx[0] = x[ m*natom_mol + i ][0];
                mx[1] = x[ m*natom_mol + i ][1];
                mx[2] = x[ m*natom_mol + i ][2];

                // Move m site to TIP4P distance if model is E3B3 or TIP4P2005 -- this must be done to use the TIP4P map
                if ( i == 3 )
                {
                    if ( model != 0 ) 
                    {
                        // get the OM unit vector
                        mom[0] = minImage( mx[0] - mox[0], boxx );
                        mom[1] = minImage( mx[1] - mox[1], boxy );
                        mom[2] = minImage( mx[2] - mox[2], boxz );
                        r      = mag3(mom);

                        // TIP4P OM distance is 0.015 nm along the OM bond
                        mx[0] = mox[0] + 0.0150*mom[0]/r;
                        mx[1] = mox[1] + 0.0150*mom[1]/r;
                        mx[2] = mox[2] + 0.0150*mom[2]/r;
                    }
                }

                // the minimum image displacement between the reference hydrogen and the current atom
                // NOTE: this converted to bohr so the efield will be in au
                dr[0]  = minImage( nhx[0] - mx[0], boxx )*bohr_nm;
                dr[1]  = minImage( nhx[1] - mx[1], boxy )*bohr_nm;
                dr[2]  = minImage( nhx[2] - mx[2], boxz )*bohr_nm;
                r      = mag3(dr);

                // Add the contribution of the current atom to the electric field
                if ( i < 3  ){              // HW1 and HW2
                    for ( j=0; j < DIM; j++){
                        efield[j] += 0.52 * dr[j] / (r*r*r);
                    }
                }
                else if ( i == 3 ){         // MW (note the negative sign)
                    for ( j=0; j < DIM; j++){
                        efield[j] -= 1.04 * dr[j] / (r*r*r);
                    }
                }
            } // end loop over atoms in molecule m

        } // end loop over molecules m

        // project the efield along the OH bond to get the relevant value for the map
        eproj[chrom] = dot3( efield, nohx );

    } // end loop over reference chromophores
}

/**********************************************************
   
   BUILD HAMILTONIAN AND RETURN TRANSITION DIPOLE VECTOR
    FOR EACH CHROMOPHORE ON THE GPU

 **********************************************************/
void get_kappa_sparse( rvec *x, float boxx, float boxy, float boxz, int natoms, int natom_mol, 
                       int nchrom, int nchrom_mol, int nmol, user_real_t *eproj, user_real_t *kappa, 
                       user_complex_t *mux, user_complex_t *muy, user_complex_t *muz, user_real_t avef, 
                       long long int *kappaFill, long long int kappaFillMax, int *kappai, int *kappaj,
                       int *nlisti, int *nlistj, long long int nlistFill )
{
    
    int n, m;
    int chromn, chromm;
    int nchroms, mchroms;
    user_real_t mox[DIM];                         // oxygen position on molecule m
    user_real_t mhx[DIM];                         // atom position on molecule m
    user_real_t nhx[DIM];                         // hydrogen position on molecule n of the current chromophore
    user_real_t nox[DIM];                         // oxygen position on molecule n
    user_real_t noh[DIM];
    user_real_t moh[DIM];
    user_real_t nmu[DIM];
    user_real_t mmu[DIM];
    user_real_t mmuprime;
    user_real_t nmuprime;
    user_real_t dr[DIM];                          // the min image vector between two atoms
    user_real_t r;                                // the distance between two atoms 
    const user_real_t bohr_nm    = 18.8973;       // convert from bohr to nanometer
    const user_real_t cm_hartree = 2.1947463E5;   // convert from cm-1 to hartree
    user_real_t En, Em;                           // the electric field projection
    user_real_t xn, xm, pn, pm;                   // the x and p from the map
    user_real_t wn, wm;                           // the energies
    user_real_t dipoleCpl;                        // transition dipole coupling


    // initialize the fill
    *kappaFill = 0;

    // Loop over molecules in neighbor list
    for ( int pair = 0; pair < nlistFill; pair ++ )
    {
        //printf("%d %lld\n", pair, nlistFill);
        // get molecule numbers for current pair
        n = nlisti[pair];
        m = nlistj[pair];
        // %printf("%d %d\n", n,m);

        // loop over chromophores on molecule n and m
        for ( nchroms = 0; nchroms<nchrom_mol; nchroms++ )
        {
            // get chromophore number
            chromn = n*nchrom_mol + nchroms;

            // get the corresponding electric field at the relevant hydrogen
            En  = eproj[chromn];

            // build the map
            wn  = 3760.2 - 3541.7*En - 152677.0*En*En;
            xn  = 0.19285 - 1.7261E-5 * wn;
            pn  = 1.6466  + 5.7692E-4 * wn;
            nmuprime = 0.1646 + 11.39*En + 63.41*En*En;

            // and calculate the location of the transition dipole moment
            // See calc_efield for assumptions about ordering of atoms
            nox[0]  = x[ n*natom_mol ][0];
            nox[1]  = x[ n*natom_mol ][1];
            nox[2]  = x[ n*natom_mol ][2];
            if ( chromn % 2 == 0 )       //HW1
            {
                nhx[0]  = x[ n*natom_mol + 1 ][0];
                nhx[1]  = x[ n*natom_mol + 1 ][1];
                nhx[2]  = x[ n*natom_mol + 1 ][2];
            }
            else if ( chromn % 2 == 1 )  //HW2
            {
                nhx[0]  = x[ n*natom_mol + 2 ][0];
                nhx[1]  = x[ n*natom_mol + 2 ][1];
                nhx[2]  = x[ n*natom_mol + 2 ][2];
            }

            // The OH unit vector
            noh[0] = minImage( nhx[0] - nox[0], boxx );
            noh[1] = minImage( nhx[1] - nox[1], boxy );
            noh[2] = minImage( nhx[2] - nox[2], boxz );
            r      = mag3(noh);
            noh[0] /= r;
            noh[1] /= r;
            noh[2] /= r;

            // The location of the TDM
            nmu[0] = minImage( nox[0] + 0.067 * noh[0], boxx );
            nmu[1] = minImage( nox[1] + 0.067 * noh[1], boxy );
            nmu[2] = minImage( nox[2] + 0.067 * noh[2], boxz );
        
            // and the TDM vector to return (make complex)
            mux[chromn] = (1. + 0*I) * noh[0] * nmuprime * xn;
            muy[chromn] = (1. + 0*I) * noh[1] * nmuprime * xn;
            muz[chromn] = (1. + 0*I) * noh[2] * nmuprime * xn;



            // Loop over chromophores on molecule m
            for ( mchroms = 0; mchroms<nchrom_mol; mchroms++ )
            {
                chromm = m*nchrom_mol + mchroms;
                if (chromm < chromn) continue; // only do upper diagonal
 
                // get the corresponding electric field at the relevant hydrogen
                Em  = eproj[chromm];

                // also get the relevent x and p from the map
                wm  = 3760.2 - 3541.7*Em - 152677.0*Em*Em;
                xm  = 0.19285 - 1.7261E-5 * wm;
                pm  = 1.6466  + 5.7692E-4 * wm;
                mmuprime = 0.1646 + 11.39*Em + 63.41*Em*Em;

                // the diagonal energy
                if ( chromn == chromm )
                {
                    // Note that this is a flattened 2d array -- subtract high frequency energies to get rid of highly oscillatory parts of the F matrix
                    kappai[*kappaFill] = chromn; 
                    kappaj[*kappaFill] = chromm;
                    kappa [*kappaFill] = wm - avef;
                    *kappaFill += 1;
                }

                // intramolecular coupling
                else if ( m == n )
                {
                    kappai[*kappaFill] = chromn; 
                    kappaj[*kappaFill] = chromm;
                    kappa [*kappaFill] = (-1361.0 + 27165*(En + Em))*xn*xm - 1.887*pn*pm;
                    *kappaFill += 1;
                }

                // intermolecular coupling
                else
                {
                
                    // calculate the distance between dipoles
                    // they are located 0.67 A from the oxygen along the OH bond
                    // tdm position on chromophore n
                    mox[0]  = x[ m*natom_mol ][0];
                    mox[1]  = x[ m*natom_mol ][1];
                    mox[2]  = x[ m*natom_mol ][2];
                    if ( chromm % 2 == 0 )       //HW1
                    {
                        mhx[0]  = x[ m*natom_mol + 1 ][0];
                        mhx[1]  = x[ m*natom_mol + 1 ][1];
                        mhx[2]  = x[ m*natom_mol + 1 ][2];
                    }
                    else if ( chromm % 2 == 1 )  //HW2
                    {
                        mhx[0]  = x[ m*natom_mol + 2 ][0];
                        mhx[1]  = x[ m*natom_mol + 2 ][1];
                        mhx[2]  = x[ m*natom_mol + 2 ][2];
                    }

                    // The OH unit vector
                    moh[0] = minImage( mhx[0] - mox[0], boxx );
                    moh[1] = minImage( mhx[1] - mox[1], boxy );
                    moh[2] = minImage( mhx[2] - mox[2], boxz );
                    r      = mag3(moh);
                    moh[0] /= r;
                    moh[1] /= r;
                    moh[2] /= r;

                    // The location of the TDM and the dipole derivative
                    mmu[0] = minImage( mox[0] + 0.067 * moh[0], boxx );
                    mmu[1] = minImage( mox[1] + 0.067 * moh[1], boxy );
                    mmu[2] = minImage( mox[2] + 0.067 * moh[2], boxz );

                    // the distance between TDM on N and on M and convert to unit vector
                    dr[0] = minImage( nmu[0] - mmu[0], boxx );
                    dr[1] = minImage( nmu[1] - mmu[1], boxy );
                    dr[2] = minImage( nmu[2] - mmu[2], boxz );
                    r     = mag3( dr );
                    dr[0] /= r;
                    dr[1] /= r;
                    dr[2] /= r;
                    r     *= bohr_nm; // convert to bohr

                    // calculate coupling in wavenumber
                    dipoleCpl = ( dot3( noh, moh ) - 3.0 * dot3( noh, dr ) * 
                                             dot3( moh, dr ) ) / ( r*r*r ) * 
                                         xn*xm*nmuprime*mmuprime*cm_hartree;

                    // only keep it if it is greater than 1 wavenumber
                    if ( fabs(dipoleCpl) > 1. )
                    {
                        kappai[*kappaFill] = chromn; 
                        kappaj[*kappaFill] = chromm;
                        kappa [*kappaFill] = dipoleCpl;
                        *kappaFill += 1 ;
                    }

                }// end intramolecular coupling

                if (*kappaFill >= kappaFillMax) {
                    printf("kappaFill >= kappaFillMax (=%lld). Aborting!\n", kappaFillMax );
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
 
}



/**********************************************************
   
        HELPER FUNCTIONS FOR GPU CALCULATIONS
            CALLABLE FROM CPU AND GPU

 **********************************************************/



// The minimage image of a scalar
user_real_t minImage( user_real_t dx, user_real_t boxl )
{
    return dx - boxl*round(dx/boxl);
}



// The magnitude of a 3 dimensional vector
user_real_t mag3( user_real_t dx[3] )
{
    return sqrt( dot3( dx, dx ) );
}



// The dot product of a 3 dimensional vector
user_real_t dot3( user_real_t x[3], user_real_t y[3] )
{
    return  x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}



// parse input file to setup calculation
void ir_init( char *argv[], char gmxf[], char cptf[], char outf[], char model[], user_real_t *dt, int *ntcfpoints, 
              int *nsamples, int *sampleEvery, user_real_t *t1, user_real_t *avef, int *natom_mol, int *nchrom_mol, 
              int *nzeros, user_real_t *beginTime, user_real_t *cplCut, int *maxCouple, int rank )
{
    char                para[MAX_STR_LEN];
    char                value[MAX_STR_LEN];

    FILE *inpf = fopen(argv[1],"r");
    if ( inpf == NULL )
    {
        printf("ERROR: Could not open %s. The first argument should contain  a  vaild\nfile name that points to a file containing the simulation parameters.\n", argv[1]);
        MPI_Finalize();
        exit(0);
    }
    else mpiprintf(">>> Reading parameters from input file %s\n", argv[1]);

    // Parse input file
    while (fscanf( inpf, "%s%s%*[^\n]", para, value ) != EOF)
    {
        if ( strcmp(para,"xtcf") == 0 ) 
        {
            sscanf( value, "%s", gmxf );
        }
        else if ( strcmp(para,"outf") == 0 )
        {
            sscanf( value, "%s", outf );
        }
        else if ( strcmp(para,"cptf") == 0 ) 
        {
            sscanf( value, "%s", cptf );
        }
        else if ( strcmp(para,"model") == 0 )
        {
            sscanf( value, "%s", model );
        }
        else if ( strcmp(para,"ntcfpoints") == 0 )
        {
            sscanf( value, "%d", (int *) ntcfpoints );
        }
        else if ( strcmp(para,"nsamples") == 0 )
        {
            sscanf( value, "%d", (int *) nsamples);
        }
        else if ( strcmp(para,"sampleEvery") == 0 )
        {
            sscanf( value, "%d", (int *) sampleEvery );
        }
        else if ( strcmp(para,"natom_mol") == 0 )
        {
            sscanf( value, "%d", (int *) natom_mol );
        }
        else if ( strcmp(para,"nchrom_mol") == 0 )
        {
            sscanf( value, "%d", (int *) nchrom_mol );
        }
        else if ( strcmp(para,"nzeros") == 0 )
        {
            sscanf( value, "%d", (int *) nzeros );
        }
        else if ( strcmp(para,"maxCouple") == 0 )
        {
            sscanf( value, "%d", (int *) maxCouple );
        }
#ifdef USE_DOUBLES
        else if ( strcmp(para,"dt") == 0 )
        {
            sscanf( value, "%lf", dt );
        }
        else if ( strcmp(para,"t1") == 0 )
        {
            sscanf( value, "%lf", t1 );
        }
        else if ( strcmp(para,"avef") == 0 )
        {
            sscanf( value, "%lf", avef );
        }
        else if ( strcmp(para,"beginTime") == 0 )
        {
            sscanf( value, "%lf", beginTime );
        }
        else if ( strcmp(para,"cplCut") == 0 )
        {
            sscanf( value, "%lf", cplCut );
        }
#else
        else if ( strcmp(para,"dt") == 0 )
        {
            sscanf( value, "%f", dt );
        }
        else if ( strcmp(para,"t1") == 0 )
        {
            sscanf( value, "%f", t1 );
        }
        else if ( strcmp(para,"avef") == 0 )
        {
            sscanf( value, "%f", avef );
        }
        else if ( strcmp(para,"beginTime") == 0 )
        {
            sscanf( value, "%f", beginTime );
        }
        else if ( strcmp(para,"cplCut") == 0 )
        {
            sscanf( value, "%f", cplCut );
        }
#endif
        else
        {
            mpiprintf("WARNING: Parameter %s in input file %s not recognized, ignoring.\n", para, argv[1]);
        }
    }

    fclose(inpf);
    mpiprintf(">>> Done reading input file and setting parameters\n");

}



// Progress bar to keep updated on tcf
void printProgress( int currentStep, int totalSteps )
{
    user_real_t percentage = (user_real_t) currentStep / (user_real_t) totalSteps;
    int lpad = (int) (percentage*PWID);
    int rpad = PWID - lpad;
    fprintf(stderr, "\r [%.*s%*s]%3d%%", lpad, PSTR, rpad, "",(int) (percentage*100));
}
