/*****************************************************************************

  Ver. 3.9 (26/July/2019)

  OpenMX (Open source package for Material eXplorer) is a program package
  for linear scaling density functional calculations of large-scale materials.
  Almost time-consuming parts can be performed in O(N) operations where N is
  the number of atoms (or basis orbitals). Thus, the program might be useful
  for studies of large-scale materials.
  The distribution of this program package follows the practice of
  the GNU General Public Licence (GPL).

  OpenMX is based on 

   *  Local density and generalized gradient approximation (LDA, LSDA, GGA)
      to the exchange-corellation term
   *  Norm-conserving pseudo potentials
   *  Variationally optimized pseudo atomic basis orbitals
   *  Solution of Poisson's equation using FFT
   *  Evaluation of two-center integrals using Fourier transformation
   *  Evaluation of three-center integrals using fixed real space grids
   *  Simple mixing, direct inversion in the interative subspace (DIIS),
      and Guaranteed-reduction Pulay's methods for SCF calculations.
   *  Solution of the eigenvalue problem using O(N) methods
   *  ...

  See also our website (http://www.openmx-square.org/)
  for recent developments.


    **************************************************************
     Copyright

     Taisuke Ozaki

     Present (23/Sep./2019) official address

       Institute for Solid State Physics, University of Tokyo,
       Kashiwanoha 5-1-5, Kashiwa, Chiba 277-8581, Japan

       e-mail: t-ozaki@issp.u-tokyo.ac.jp
    **************************************************************
 
*****************************************************************************/

/**********************************************************************
  openmx.c:

     openmx.c is the main routine of OpenMX.

  Log of openmx.c:

     5/Oct/2003  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
/*  end stat section */
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#include "tran_prototypes.h"
#include "tran_variables.h"

#include "hdf5.h"

void Make_VNA_Grid();
void set_O_nm(double ****O);
void output_O_nm(double ****O, const char *output_name, int ParallelHDF5, double coefficient);

    int main(int argc, char *argv[])
{ 
  static int numprocs,myid;
  static int MD_iter,i,j,po,ip;
  static char fileMemory[YOUSO10]; 
  double TStime,TEtime;
  MPI_Comm mpi_comm_parent;

  /* MPI initialize */

  mpi_comm_level1 = MPI_COMM_WORLD; 
  MPI_COMM_WORLD1 = MPI_COMM_WORLD; 

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD1,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);
  NUMPROCS_MPI_COMM_WORLD = numprocs;
  MYID_MPI_COMM_WORLD = myid;
  Num_Procs = numprocs;

  /* check if OpenMX was called by MPI_spawn. */

  MPI_Comm_get_parent(&mpi_comm_parent);
  if (mpi_comm_parent!=MPI_COMM_NULL) MPI_spawn_flag = 1;
  else                                MPI_spawn_flag = 0;

  /* for measuring elapsed time */

  dtime(&TStime);

  /* check argv */

  if (argc==1){

    if (myid==Host_ID) printf("\nCould not find an input file.\n\n");
    MPI_Finalize(); 
    exit(0);
  } 

  /* initialize Runtest_flag */

  Runtest_flag = 0;

  /****************************************************
    ./openmx -nt # 

    specifies the number of threads in parallelization
    by OpenMP
  ****************************************************/
  
  openmp_threads_num = 1; /* default */

  po = 0;
  if (myid==Host_ID){
    for (i=0; i<argc; i++){
      if ( strcmp(argv[i],"-nt")==0 ){
        po = 1;
        ip = i;
      }
    }
  }

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);
  MPI_Bcast(&ip, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);

  if ( (argc-1)<(ip+1) ){
    if (myid==Host_ID){
      printf("cannot find the number of threads\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if ( po==1 ){
    openmp_threads_num = atoi(argv[ip+1]);

    if (openmp_threads_num<=0){ 
      if (myid==Host_ID){
        printf("check the number of threads\n");
      }
      MPI_Finalize();
      exit(0);
    }
  }

  omp_set_num_threads(openmp_threads_num);  

  if (myid==Host_ID){
    printf("\nThe number of threads in each node for OpenMP parallelization is %d.\n\n",openmp_threads_num);
  }

  /****************************************************
    ./openmx -show directory 

    showing PAOs and VPS used in input files stored in 
    "directory".
  ****************************************************/

  if (strcmp(argv[1],"-show")==0){
    Show_DFT_DATA(argv);
    exit(0);
  }

  /****************************************************
    ./openmx -maketest

    making of *.out files in order to check whether 
    OpenMX normally runs on many platforms or not.
  ****************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketest")==0){
    Maketest("S",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtest

   check whether OpenMX normally runs on many platforms
   or not by comparing the stored *.out and generated
   *.out on your machine.
  ****************************************************/

  if ( strcmp(argv[1],"-runtest")==0){
    Runtest("S",argc,argv);
  }

  /****************************************************
   ./openmx -maketestL

    making of *.out files in order to check whether 
    OpenMX normally runs for relatively large systems
    on many platforms or not
  ****************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestL")==0){
    Maketest("L",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -maketestL2

    making of *.out files in order to check whether 
    OpenMX normally runs for large systems on many 
    platforms or not
  ****************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestL2")==0){
    Maketest("L2",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -maketestL3

    making of *.out files in order to check whether 
    OpenMX normally runs for large systems on many 
    platforms or not
  ****************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestL3")==0){
    Maketest("L3",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestL

   check whether OpenMX normally runs for relatively
   large systems on many platforms or not by comparing
   the stored *.out and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1],"-runtestL")==0){
    Runtest("L",argc,argv);
  }

  /****************************************************
   ./openmx -runtestL2

   check whether OpenMX normally runs for large systems 
   on many platforms or not by comparing the stored *.out 
   and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1],"-runtestL2")==0){
    Runtest("L2",argc,argv);
  }

  /****************************************************
   ./openmx -runtestL3

   check whether OpenMX normally runs for small, medium size, 
   and large systems on many platforms or not by comparing 
   the stored *.out and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1],"-runtestL3")==0){
    Runtest("L3",argc,argv);
  }

  /*******************************************************
   check memory leak by monitoring actual used memory size
  *******************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-mltest")==0){
    Memory_Leak_test(argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -maketestG

    making of *.out files in order to check whether 
    OpenMX normally runs for geometry optimization
    on many platforms or not
  ****************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestG")==0){
    Maketest("G",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -maketestC

    making of *.out files in order to check whether 
    OpenMX normally runs for geometry optimization
    on many platforms or not
  ****************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestC")==0){
    Maketest("C",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestG

   check whether OpenMX normally runs for geometry 
   optimization on many platforms or not by comparing
   the stored *.out and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1],"-runtestG")==0){
    Runtest("G",argc,argv);
  }

  /****************************************************
   ./openmx -runtestC

   check whether OpenMX normally runs for simultaneous 
   optimization for cell and geometry on many platforms 
   or not by comparing the stored *.out and generated
   *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1],"-runtestC")==0){
    Runtest("C",argc,argv);
  }

  /****************************************************
   ./openmx -maketestWF

    making of *.out files in order to check whether 
    OpenMX normally runs for generation of Wannier 
    functions on many platforms or not
  ****************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestWF")==0){
    Maketest("WF",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestWF

   check whether OpenMX normally runs for generating 
   Wannier functions on many platforms or not by comparing
   the stored *.out and generated *.out on your machine.
  ****************************************************/

  if (strcmp(argv[1],"-runtestWF")==0){
    Runtest("WF",argc,argv);
  }

  /****************************************************
   ./openmx -maketestNEGF

    making of *.out files in order to check whether 
    OpenMX normally runs for NEGF calculations 
    on many platforms or not
  ****************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestNEGF")==0){
    Maketest("NEGF",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestNEGF

    check whether OpenMX normally runs for NEGF calculations 
    on many platforms or not
  ****************************************************/

  if (strcmp(argv[1],"-runtestNEGF")==0){
    Runtest("NEGF",argc,argv);
    MPI_Finalize();
    exit(0);
  }

  /********************************************************************
   ./openmx -maketestCDDF

    making of *.df_re and *.df_im files in order to check whether
    OpenMX normally runs for CDDF calculations on many platforms or not
  *********************************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestCDDF")==0){
    Maketest("CDDF",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestCDDF

    check whether OpenMX normally runs for CDDF calculations 
    on many platforms or not
  ****************************************************/

  if (strcmp(argv[1],"-runtestCDDF")==0){
    Runtest("CDDF",argc,argv);
    MPI_Finalize();
    exit(0);
  }

  /********************************************************************
   ./openmx -maketestDCLNO

    making of *.out files in order to check whether 
    OpenMX normally runs for DC-LNO calculations
    on many platforms or not
  *********************************************************************/

  if ( (argc==2 || argc==3) && strcmp(argv[1],"-maketestDCLNO")==0){
    Maketest("DCLNO",argc,argv);
    exit(0);
  }

  /****************************************************
   ./openmx -runtestDCLNO

   check whether OpenMX normally runs for DCLNO calculations 
   on many platforms or not
  ****************************************************/

  if (strcmp(argv[1],"-runtestDCLNO")==0){
    Runtest("DCLNO",argc,argv);
    MPI_Finalize();
    exit(0);
  }

  /*******************************************************
   check consistency between analytic and numerical forces
  *******************************************************/

  if ( (argc==3 || argc==4) && strcmp(argv[1],"-forcetest")==0){

    if      (strcmp(argv[2],"0")==0) force_flag = 0; 
    else if (strcmp(argv[2],"1")==0) force_flag = 1; 
    else if (strcmp(argv[2],"2")==0) force_flag = 2; 
    else if (strcmp(argv[2],"3")==0) force_flag = 3; 
    else if (strcmp(argv[2],"4")==0) force_flag = 4; 
    else if (strcmp(argv[2],"5")==0) force_flag = 5; 
    else if (strcmp(argv[2],"6")==0) force_flag = 6; 
    else if (strcmp(argv[2],"7")==0) force_flag = 7;
    else if (strcmp(argv[2],"8")==0) force_flag = 8;
    else {
      printf("unsupported flag for -forcetest\n");
      exit(0);
    }

    Force_test(argc,argv);
    exit(0);
  }

  /*********************************************************
   check consistency between analytic and numerical stress
  *********************************************************/

  if ( (argc==3 || argc==4) && strcmp(argv[1],"-stresstest")==0){

    if      (strcmp(argv[2],"0")==0) stress_flag = 0; 
    else if (strcmp(argv[2],"1")==0) stress_flag = 1; 
    else if (strcmp(argv[2],"2")==0) stress_flag = 2; 
    else if (strcmp(argv[2],"3")==0) stress_flag = 3; 
    else if (strcmp(argv[2],"4")==0) stress_flag = 4; 
    else if (strcmp(argv[2],"5")==0) stress_flag = 5; 
    else if (strcmp(argv[2],"6")==0) stress_flag = 6; 
    else if (strcmp(argv[2],"7")==0) stress_flag = 7;
    else if (strcmp(argv[2],"8")==0) stress_flag = 8;
    else {
      printf("unsupported flag for -stresstest\n");
      exit(0);
    }

    Stress_test(argc,argv);
    MPI_Finalize();
    exit(0);
  }

  /*******************************************************
    check the NEB calculation or not, and if yes, go to 
    the NEB calculation.
  *******************************************************/

  if (neb_check(argv)) neb(argc,argv);

  /*******************************************************
   allocation of CompTime and show the greeting message 
  *******************************************************/

  CompTime = (double**)malloc(sizeof(double*)*numprocs); 
  for (i=0; i<numprocs; i++){
    CompTime[i] = (double*)malloc(sizeof(double)*30); 
    for (j=0; j<30; j++) CompTime[i][j] = 0.0;
  }

  if (myid==Host_ID){  
    printf("\n*******************************************************\n"); 
    printf("*******************************************************\n"); 
    printf(" Welcome to OpenMX   Ver. %s                           \n",Version_OpenMX); 
    printf(" Copyright (C), 2002-2019, T. Ozaki                    \n"); 
    printf(" OpenMX comes with ABSOLUTELY NO WARRANTY.             \n"); 
    printf(" This is free software, and you are welcome to         \n"); 
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("*******************************************************\n"); 
    printf("*******************************************************\n\n"); 
  } 

  Init_List_YOUSO();
  remake_headfile = 0;
  ScaleSize = 1.2; 

  /****************************************************
                   Read the input file
  ****************************************************/

  init_alloc_first();

  CompTime[myid][1] = readfile(argv);
  MPI_Barrier(MPI_COMM_WORLD1);

  /* initialize PrintMemory routine */

  sprintf(fileMemory,"%s%s.memory%i",filepath,filename,myid);
  PrintMemory(fileMemory,0,"init"); 
  PrintMemory_Fix();

  /* initialize */

  init();

  /* check "-mltest2" mode */

  po = 0;
  if (myid==Host_ID){
    for (i=0; i<argc; i++){
      if ( strcmp(argv[i],"-mltest2")==0 ){
        po = 1;
        ip = i;
      }
    }
  }

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);
  MPI_Bcast(&ip, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);

  if ( po==1 ) ML_flag = 1;
  else         ML_flag = 0;

  /* check "-forcetest2" mode */

  po = 0;
  if (myid==Host_ID){
    for (i=0; i<argc; i++){
      if ( strcmp(argv[i],"-forcetest2")==0 ){
        po = 1;
        ip = i;
      }
    }
  }

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);
  MPI_Bcast(&ip, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);

  if ( po==1 ){
    force_flag = atoi(argv[ip+1]);
    ForceConsistency_flag = 1;
  }

  /* check force consistency 
     the number of processes 
     should be less than 2.
  */

  if (ForceConsistency_flag==1){

    Check_Force(argv);
    CompTime[myid][20] = OutData(argv[1]);
    Merge_LogFile(argv[1]);
    Free_Arrays(0);
    MPI_Finalize();
    exit(0); 
    return 0;
  }

  /* check "-stresstest2" mode */

  po = 0;
  if (myid==Host_ID){


    for (i=0; i<argc; i++){
      if ( strcmp(argv[i],"-stresstest2")==0 ){

        po = 1;
        ip = i;
      }
    }
  }

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);
  MPI_Bcast(&ip, 1, MPI_INT, Host_ID, MPI_COMM_WORLD1);

  if ( po==1 ){
    stress_flag = atoi(argv[ip+1]);
    StressConsistency_flag = 1;
  }

  /* check stress consistency 
     the number of processes 
     should be less than 2.
  */

  if (StressConsistency_flag==1){
    Check_Stress(argv);
    CompTime[myid][20] = OutData(argv[1]);
    Merge_LogFile(argv[1]);
    Free_Arrays(0);
    MPI_Finalize();
    exit(0); 
    return 0;
  }

    /****************************************************
         Calculation and output for OLP T VNL (VNA)
    ****************************************************/
    MD_iter = 1;

    // allocation of arrays


//    if (MD_switch==12)
//        CompTime[myid][2] += truncation(1,1);  /* EvsLC */
//    else if (MD_cellopt_flag==1)
//        CompTime[myid][2] += truncation(1,1);  /* cell optimization */
//    else
//        CompTime[myid][2] += truncation(MD_iter,1);

    double time1, time2;
    if (ML_flag==1 && myid==Host_ID) Get_VSZ(MD_iter);
    const double Hartree2Ev = 27.2113845;
    truncation(1, 0);

    if (myid==Host_ID)
    {
        printf("\n Calc. OLP ...\n");
        mkdir("output", 0775);
    }
    time1 = Set_OLP_Kin(OLP, H0);
    output_O_nm(OLP[0], "output/overlaps", 0, 1.0);
    if (myid==Host_ID)
        printf("\n Finish calc. OLP\n");


  MPI_Barrier(MPI_COMM_WORLD1);
  if (myid==Host_ID){
    printf("\nThe calculation was normally finished.\n");fflush(stdout);
  }

  Make_FracCoord(argv[1]);
  Merge_LogFile(argv[1]);

  /* if OpenMX is called by MPI_spawn. */

  if (MPI_spawn_flag==1){

    MPI_Comm_get_parent(&mpi_comm_parent);

    if(mpi_comm_parent!=MPI_COMM_NULL){
      //MPI_Comm_disconnect(&mpi_comm_parent);
    }

    fclose(MPI_spawn_stream);

    MPI_Finalize();
  }   
  else{
    MPI_Finalize();
    exit(0);
  }

  return 0;
}

void Make_VNA_Grid()
{
    static int firsttime=1;
    unsigned long long int n2D,N2D,GNc,GN;
    int i,Mc_AN,Gc_AN,BN,CN,LN,GRc,N3[4];
    int AN,Nc,MN,Cwan,NN_S,NN_R;
    int size_AtomVNA_Grid;
    int size_AtomVNA_Snd_Grid_A2B;
    int size_AtomVNA_Rcv_Grid_A2B;
    double Cxyz[4];
    double r,dx,dy,dz;
    double **AtomVNA_Grid;
    double **AtomVNA_Snd_Grid_A2B;
    double **AtomVNA_Rcv_Grid_A2B;
    double Stime_atom, Etime_atom;
    int numprocs,myid,tag=999,ID,IDS,IDR;
    int OMPID,Nthrds,Nprocs;

    MPI_Status stat;
    MPI_Request request;
    MPI_Status *stat_send;
    MPI_Status *stat_recv;
    MPI_Request *request_send;
    MPI_Request *request_recv;

    /* MPI */
    MPI_Comm_size(mpi_comm_level1,&numprocs);
    MPI_Comm_rank(mpi_comm_level1,&myid);

    /* allocation of arrays */

    size_AtomVNA_Grid = 1;
    AtomVNA_Grid = (double**)malloc(sizeof(double*)*(Matomnum+1));
    AtomVNA_Grid[0] = (double*)malloc(sizeof(double)*1);
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = F_M2G[Mc_AN];
        AtomVNA_Grid[Mc_AN] = (double*)malloc(sizeof(double)*GridN_Atom[Gc_AN]);
        size_AtomVNA_Grid += GridN_Atom[Gc_AN];
    }

    size_AtomVNA_Snd_Grid_A2B = 0;
    AtomVNA_Snd_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
    for (ID=0; ID<numprocs; ID++){
        AtomVNA_Snd_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Snd_Grid_A2B[ID]);
        size_AtomVNA_Snd_Grid_A2B += Num_Snd_Grid_A2B[ID];
    }

    size_AtomVNA_Rcv_Grid_A2B = 0;
    AtomVNA_Rcv_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
    for (ID=0; ID<numprocs; ID++){
        AtomVNA_Rcv_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Rcv_Grid_A2B[ID]);
        size_AtomVNA_Rcv_Grid_A2B += Num_Rcv_Grid_A2B[ID];
    }

    /* PrintMemory */
    if (firsttime) {
        PrintMemory("Set_Vpot: AtomVNA_Grid",sizeof(double)*size_AtomVNA_Grid,NULL);
        PrintMemory("Set_Vpot: AtomVNA_Snd_Grid_A2B",sizeof(double)*size_AtomVNA_Snd_Grid_A2B,NULL);
        PrintMemory("Set_Vpot: AtomVNA_Rcv_Grid_A2B",sizeof(double)*size_AtomVNA_Rcv_Grid_A2B,NULL);
    }

    /* calculation of AtomVNA_Grid */

    for (MN=0; MN<My_NumGridC; MN++) VNA_Grid[MN] = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];

#pragma omp parallel shared(AtomVNA_Grid,GridN_Atom,atv,Gxyz,Gc_AN,Cwan,Mc_AN,GridListAtom,CellListAtom) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,dx,dy,dz,r)
        {

            OMPID = omp_get_thread_num();
            Nthrds = omp_get_num_threads();
            Nprocs = omp_get_num_procs();

            for (Nc=OMPID*GridN_Atom[Gc_AN]/Nthrds; Nc<(OMPID+1)*GridN_Atom[Gc_AN]/Nthrds; Nc++){

                GNc = GridListAtom[Mc_AN][Nc];
                GRc = CellListAtom[Mc_AN][Nc];

                Get_Grid_XYZ(GNc,Cxyz);
                dx = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1];
                dy = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2];
                dz = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

                r = sqrt(dx*dx + dy*dy + dz*dz);
                AtomVNA_Grid[Mc_AN][Nc] = VNAF(Cwan,r);
            }

#pragma omp flush(AtomVNA_Grid)

        } /* #pragma omp parallel */

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /******************************************************
      MPI communication from the partitions A to B
    ******************************************************/

    /* copy AtomVNA_Grid to AtomVNA_Snd_Grid_A2B */

    for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;

    N2D = Ngrid1*Ngrid2;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

        Gc_AN = M2G[Mc_AN];

        for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

            GN = GridListAtom[Mc_AN][AN];
            GN2N(GN,N3);
            n2D = N3[1]*Ngrid2 + N3[2];
            ID = (int)(n2D*(unsigned long long int)numprocs/N2D);
            AtomVNA_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] = AtomVNA_Grid[Mc_AN][AN];

            Num_Snd_Grid_A2B[ID]++;
        }
    }

    /* MPI: A to B */

    request_send = malloc(sizeof(MPI_Request)*NN_A2B_S);
    request_recv = malloc(sizeof(MPI_Request)*NN_A2B_R);
    stat_send = malloc(sizeof(MPI_Status)*NN_A2B_S);
    stat_recv = malloc(sizeof(MPI_Status)*NN_A2B_R);

    NN_S = 0;
    NN_R = 0;

    tag = 999;
    for (ID=1; ID<numprocs; ID++){

        IDS = (myid + ID) % numprocs;
        IDR = (myid - ID + numprocs) % numprocs;

        if (Num_Snd_Grid_A2B[IDS]!=0){
            MPI_Isend(&AtomVNA_Snd_Grid_A2B[IDS][0], Num_Snd_Grid_A2B[IDS], MPI_DOUBLE,
                      IDS, tag, mpi_comm_level1, &request_send[NN_S]);
            NN_S++;
        }

        if (Num_Rcv_Grid_A2B[IDR]!=0){
            MPI_Irecv( &AtomVNA_Rcv_Grid_A2B[IDR][0], Num_Rcv_Grid_A2B[IDR],
                       MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
            NN_R++;
        }
    }

    if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
    if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

    free(request_send);
    free(request_recv);
    free(stat_send);
    free(stat_recv);

    /* for myid */
    for (i=0; i<Num_Rcv_Grid_A2B[myid]; i++){
        AtomVNA_Rcv_Grid_A2B[myid][i] = AtomVNA_Snd_Grid_A2B[myid][i];
    }

    /******************************************************
             superposition of VNA in the partition B
    ******************************************************/

    /* initialize VNA_Grid_B */

    for (BN=0; BN<My_NumGridB_AB; BN++) VNA_Grid_B[BN] = 0.0;

    /* superposition of VNA */

    for (ID=0; ID<numprocs; ID++){
        for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){

            BN = Index_Rcv_Grid_A2B[ID][3*LN+0];
            VNA_Grid_B[BN] += AtomVNA_Rcv_Grid_A2B[ID][LN];

        } /* LN */
    } /* ID */

    /******************************************************
             MPI: from the partitions B to C
    ******************************************************/

    Data_Grid_Copy_B2C_1( VNA_Grid_B, VNA_Grid );

    /* freeing of arrays */

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
        free(AtomVNA_Grid[Mc_AN]);
    }
    free(AtomVNA_Grid);

    for (ID=0; ID<numprocs; ID++){
        free(AtomVNA_Snd_Grid_A2B[ID]);
    }
    free(AtomVNA_Snd_Grid_A2B);

    for (ID=0; ID<numprocs; ID++){
        free(AtomVNA_Rcv_Grid_A2B[ID]);
    }
    free(AtomVNA_Rcv_Grid_A2B);
}

void set_O_nm(double ****O)//from void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind)
{
    int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
    int Nh0,Nh1,Nh2,Nh3;
    int Nc0,Nc1,Nc2,Nc3;
    int MN0,MN1,MN2,MN3;
    int Nloop,OneD_Nloop;
    int *OneD2spin,*OneD2Mc_AN,*OneD2h_AN;
    int numprocs,myid;
    double time0,time1,time2,mflops;

    MPI_Comm_size(mpi_comm_level1,&numprocs);
    MPI_Comm_rank(mpi_comm_level1,&myid);

    /* one-dimensionalization of loops */

    Nloop = 0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            Nloop++;
        }
    }

    OneD2Mc_AN = (int*)malloc(sizeof(int)*Nloop);
    OneD2h_AN = (int*)malloc(sizeof(int)*Nloop);

    Nloop = 0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

            OneD2Mc_AN[Nloop] = Mc_AN;
            OneD2h_AN[Nloop] = h_AN;
            Nloop++;
        }
    }

    OneD_Nloop = Nloop;


    /* numerical integration */

#pragma omp parallel
    {
        int Nloop,spin,Mc_AN,h_AN,Gh_AN,Mh_AN,Hwan,NOLG;
        int Gc_AN,Cwan,NO0,NO1,spin0=-1,Mc_AN0=0;
        int i,j,Nc,MN,GNA,Nog,Nh,OMPID,Nthrds;
        int M,N,K,lda,ldb,ldc,ii,jj;
        double alpha,beta,Vpot;
        double sum0,sum1,sum2,sum3,sum4;
        double *ChiV0,*Chi1,*ChiV0_2,*C;

        /* allocation of arrays */

        /* AITUNE */
        double **AI_tmpH[4];
        {
            /* get size of temporary buffer */
            int AI_MaxNO = 0;
            int spe;
            for(spe = 0; spe < SpeciesNum; spe++){
                if(AI_MaxNO < Spe_Total_NO[spe]){ AI_MaxNO = Spe_Total_NO[spe];}
            }


            int spin;
            for (spin=0; spin<=SpinP_switch; spin++){
                AI_tmpH[spin] = (double**)malloc(sizeof(double*)*AI_MaxNO);

                int i;
                double *p = (double*)malloc(sizeof(double)*AI_MaxNO*AI_MaxNO);
                for(i = 0; i < AI_MaxNO; i++){
                    AI_tmpH[spin][i] = p;
                    p += AI_MaxNO;
                }
            }
        }
        /* AITUNE */

        /* starting of one-dimensionalized loop */

#pragma omp for schedule(static,1)  /* guided */  /* AITUNE */
        for (Nloop = 0; Nloop < OneD_Nloop; Nloop++){ /* AITUNE */

            int Mc_AN = OneD2Mc_AN[Nloop];
            int h_AN = OneD2h_AN[Nloop];
            int Gc_AN = M2G[Mc_AN];
            int Gh_AN = natn[Gc_AN][h_AN];
            int Mh_AN = F_G2M[Gh_AN];
            int Cwan = WhatSpecies[Gc_AN];
            int Hwan = WhatSpecies[Gh_AN];
            int GNA = GridN_Atom[Gc_AN];
            int NOLG = NumOLG[Mc_AN][h_AN];

            int NO0,NO1;
            NO0 = Spe_Total_NO[Cwan];
            NO1 = Spe_Total_NO[Hwan];

            /* quadrature for Hij  */

            /* AITUNE change order of loop */
            if(SpinP_switch==0){
                /* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */
                int i;
                for (i=0; i<NO0; i++){
                    int j;
                    for (j=0; j<NO1; j++){
                        AI_tmpH[0][i][j] = 0.0;
//                        AI_tmpH[0][i][j] = O[Mc_AN][h_AN][i][j];
                    }
                }

                int Nog;
                for (Nog=0; Nog<NOLG; Nog++){

                    int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
                    int MN = MGridListAtom[Mc_AN][Nc];
                    int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

                    double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];


                    if (G2ID[Gh_AN]==myid){
                        int i;
                        for (i=0; i<NO0; i++){

                            double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
                            int j;

                            for (j=0; j<NO1; j++){
                                AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
                            }
                        }

                    }else{
                        int i;
                        for (i=0; i<NO0; i++){

                            double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
                            int j;

                            for (j=0; j<NO1; j++){
                                AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
                            }
                        }
                    }

                }/* Nog */

                for (i=0; i<NO0; i++){
                    int j;
                    for (j=0; j<NO1; j++){
                        O[Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
                    }
                }
            }
        } /* Nloop */

        /* freeing of arrays */
        {
            int spin;
            for (spin=0; spin<=SpinP_switch; spin++){
                free(AI_tmpH[spin][0]);
                free(AI_tmpH[spin]);
            }
        }

    } /* pragma omp parallel */

    /* freeing of arrays */

    free(OneD2Mc_AN);
    free(OneD2h_AN);
}

void output_O_nm(double ****O, const char *output_name, int ParallelHDF5, double coefficient)
{
    int ID, myid, numprocs, Cnt_kind, spin, Gc_AN, Mc_AN, h_AN, Gh_AN, wan1, wan2, TNO1, TNO2, i, j, index_cell;
    double dx, dy, dz, r;
    double *data_out;
    MPI_Status stat;
    MPI_Request request;
    /* MPI */
    MPI_Comm_size(mpi_comm_level1, &numprocs);
    MPI_Comm_rank(mpi_comm_level1, &myid);


    hid_t file_id, dataset_id, memspace, filespace;
    hid_t plist_id;
    herr_t status, status_n;
    hsize_t dims[2];


    MPI_Barrier(mpi_comm_level1);
    if (ParallelHDF5 == 1)
    {
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        // H5Pset_fapl_mpio(plist_id, mpi_comm_level1, MPI_INFO_NULL);
        file_id = H5Fcreate(output_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    }
    else
    {
        char output_name_id[64] = {};
        sprintf(output_name_id, "%s_%d.h5", output_name, myid);
        file_id = H5Fcreate(output_name_id, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        ID = G2ID[Gc_AN];
        if (myid==ID){
            Mc_AN = F_G2M[Gc_AN];
            wan1 = WhatSpecies[Gc_AN];
            TNO1 = Spe_Total_CNO[wan1];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
                Gh_AN = natn[Gc_AN][h_AN];
                wan2 = WhatSpecies[Gh_AN];
                TNO2 = Spe_Total_CNO[wan2];
                index_cell = ncn[Gc_AN][h_AN];

                char key_str[64] = {};
                sprintf(key_str, "[%d, %d, %d, %d, %d]", atv_ijk[index_cell][1], atv_ijk[index_cell][2], atv_ijk[index_cell][3], Gc_AN, Gh_AN);

                dx = fabs(Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[index_cell][1]);
                dy = fabs(Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[index_cell][2]);
                dz = fabs(Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[index_cell][3]);
                r = sqrt(dx*dx + dy*dy + dz*dz); // Bohr

                data_out = (double*)malloc(sizeof(double) * TNO1 * TNO2);
                for (i=0; i<TNO1; ++i){
                    for (j=0; j<TNO2; ++j){
                        data_out[i * TNO2 + j] = O[Mc_AN][h_AN][i][j] * coefficient;
                    }
                }
                dims[0] = TNO1;
                dims[1] = TNO2;
                memspace = H5Screate_simple(2, dims, NULL);
                dataset_id = H5Dcreate(file_id, key_str, H5T_NATIVE_DOUBLE, memspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (ParallelHDF5 == 1)
                {
                    filespace = H5Dget_space(dataset_id);
                    plist_id = H5Pcreate(H5P_DATASET_XFER);
                    // H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
                    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data_out);
                }
                else
                {
                    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_out);
                }
                status = H5Sclose(memspace);
                status = H5Dclose(dataset_id);
                free(data_out);
            }
        }
    }
    if (ParallelHDF5 == 1)
        H5Pclose(plist_id);
    status = H5Fclose(file_id);
    MPI_Barrier(mpi_comm_level1);
}
