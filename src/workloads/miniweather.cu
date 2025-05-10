/*
Modified from https://github.com/mrnorman/miniWeather
From a CPU version to a GPU version
Add blocklevel preemption mechanism
*/
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Original Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include "workloads/scicomput.h"
#include "util/gpu_util.h"
#include "executor.h"
#include "util/common.h"

namespace SciComputeRaw {

#define SINGLE_PREC 
#define _NX 3200 
#define _NZ 1600 
#define _SIM_TIME 10 
#define _OUT_FREQ 50 
#define _DATA_SPEC 1

#ifdef SINGLE_PREC
  typedef float  real;
#else
  typedef double real;
#endif

constexpr real pi        = 3.14159265358979323846264338327;   //Pi
constexpr real grav      = 9.8;                               //Gravitational acceleration (m / s^2)
constexpr real cp        = 1004.;                             //Specific heat of dry air at constant pressure
constexpr real cv        = 717.;                              //Specific heat of dry air at constant volume
constexpr real rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
constexpr real p0        = 1.e5;                              //Standard pressure at the surface in Pascals
constexpr real C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
constexpr real gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
constexpr real xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
constexpr real zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
constexpr real hv_beta   = 0.05;    //How strong to diffuse the solution: hv_beta \in [0:1]
constexpr real cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
constexpr real max_speed = 450;     //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
constexpr int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
constexpr int sten_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
constexpr int NUM_VARS = 4;           //Number of fluid state variables
constexpr int ID_DENS  = 0;           //index for density ("rho")
constexpr int ID_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
constexpr int ID_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
constexpr int ID_RHOT  = 3;           //index for density * potential temperature ("rho * theta")
constexpr int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
constexpr int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction
constexpr int DATA_SPEC_COLLISION       = 1;
constexpr int DATA_SPEC_THERMAL         = 2;
constexpr int DATA_SPEC_GRAVITY_WAVES   = 3;
constexpr int DATA_SPEC_DENSITY_CURRENT = 5;
constexpr int DATA_SPEC_INJECTION       = 6;

constexpr int nqpoints = 3;
constexpr real qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
constexpr real qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };

///////////////////////////////////////////////////////////////////////////////////////
// BEGIN USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////
//The x-direction length is twice as long as the z-direction length
//So, you'll want to have nx_glob be twice as large as nz_glob
int constexpr nx_glob       = _NX;            //Number of total cells in the x-direction
int constexpr nz_glob       = _NZ;            //Number of total cells in the z-direction
real constexpr sim_time      = _SIM_TIME;      //How many seconds to run the simulation
real constexpr output_freq   = _OUT_FREQ;      //How frequently to output data to file (in seconds)
int constexpr data_spec_int = _DATA_SPEC;     //How to initialize the data
real constexpr dx            = xlen / nx_glob; // grid spacing in the x-direction
real constexpr dz            = zlen / nz_glob; // grid spacing in the x-direction
///////////////////////////////////////////////////////////////////////////////////////
// END USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
real dt;                    //Model time step (seconds)
constexpr int nx = nx_glob;
constexpr int nz = nz_glob;                //Number of local grid cells in the x- and z- dimensions for this MPI task
constexpr int i_beg = 0;
constexpr int k_beg = 0;          //beginning index in the x- and z-directions for this MPI task
int    nranks, myrank;        //Number of MPI ranks and my rank id
int    left_rank, right_rank; //MPI Rank IDs that exist to my left and right in the global domain
int    mainproc;            //Am I the main process (rank == 0)?

__device__ real *hy_dens_cell;         //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
__device__ real *hy_dens_theta_cell;   //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
__device__ real *hy_dens_int;          //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
__device__ real *hy_dens_theta_int;    //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
__device__ real *hy_pressure_int;      //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)
__device__ real *hy_dens_cell_const;

real *hy_dens_cell_ptr;         //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
real *hy_dens_theta_cell_ptr;   //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
real *hy_dens_int_ptr;          //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
real *hy_dens_theta_int_ptr;    //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
real *hy_pressure_int_ptr;      //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)
real *hy_dens_cell_const_ptr;
void init_device_var(){
  GPUMemAlloc((GPUdeviceptr *)&hy_dens_cell_ptr, (nz+2*hs)*sizeof(real));
  GPUMemAlloc((GPUdeviceptr *)&hy_dens_theta_cell_ptr, (nz+2*hs)*sizeof(real));
  GPUMemAlloc((GPUdeviceptr *)&hy_dens_int_ptr, (nz+1)*sizeof(real));
  GPUMemAlloc((GPUdeviceptr *)&hy_dens_theta_int_ptr, (nz+1)*sizeof(real));
  GPUMemAlloc((GPUdeviceptr *)&hy_pressure_int_ptr, (nz+1)*sizeof(real));
  GPUMemAlloc((GPUdeviceptr *)&hy_dens_cell_const_ptr, 4*sizeof(real));
  GPUMemcpyToSymbol(hy_dens_cell, &hy_dens_cell_ptr, sizeof(real*));
  GPUMemcpyToSymbol(hy_dens_theta_cell, &hy_dens_theta_cell_ptr, sizeof(real*));
  GPUMemcpyToSymbol(hy_dens_int, &hy_dens_int_ptr, sizeof(real*));
  GPUMemcpyToSymbol(hy_dens_theta_int, &hy_dens_theta_int_ptr, sizeof(real*));
  GPUMemcpyToSymbol(hy_pressure_int, &hy_pressure_int_ptr, sizeof(real*));
  GPUMemcpyToSymbol(hy_dens_cell_const, &hy_dens_cell_const_ptr, sizeof(real*));
}

real *host_hy_dens_cell;         //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
real *host_hy_dens_theta_cell;   //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
real *host_hy_dens_int;          //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
real *host_hy_dens_theta_int;    //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
real *host_hy_pressure_int;      //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
real etime;                 //Elapsed model time
real output_counter;        //Helps determine when it's time to do output

//Runtime variable arrays
real *host_state;                //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
real *host_state_tmp;            //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
real *host_flux;                 //Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)

real *state;                //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
real *state_tmp;            //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
real *flux;                 //Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)

int    num_out = 0;           //The number of outputs performed so far
int    direction_switch = 1;
double mass , te ;            //Domain totals for mass and total energy  

//How is this not in the standard?!
real dmin( real a , real b ) { if (a<b) {return a;} else {return b;} };


//Declaring the functions defined after "main"
void   init                 ( );
void   finalize             ( );
void   injection            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void   density_current      ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void   gravity_waves        ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void   thermal              ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void   collision            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void   hydro_const_theta    ( real z                   , real &r , real &t );
void   hydro_const_bvfreq   ( real z , real bv_freq0 , real &r , real &t );
real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad );
void   perform_timestep     ( real *state , real *state_tmp , real *flux , real dt, GPUstream stream);
void   semi_discrete_step   ( real *state_init , real *state_forcing , real *state_out , real dt , int dir , real *flux, GPUstream stream);
__global__ void   compute_tendencies_x ( real *state , real *flux , real dt);
__global__ void   compute_tendencies_z ( real *state , real *flux , real dt);
__global__ void   set_halo_values_x    ( real *state );
__global__ void   set_halo_values_z    ( real *state );
void   reductions           ( double &mass , double &te );

//Performs a single dimensionally split time step using a simple low-storage three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void perform_timestep(GPUstream stream) {

    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux, stream);
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux, stream);
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux, stream);

    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux, stream);
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux, stream);
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux, stream);

    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux, stream);
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux, stream);
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux, stream);

    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux, stream);
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux, stream);
    semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux, stream);

}

__global__ void compute_tendencies_x_r( real *state , real *flux , real dt, real * state_init,real *state_out);
__global__ void compute_tendencies_z_r( real *state , real *flux , real dt, real * state_init,real *state_out);

//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step( real *state_init , real *state_forcing , real *state_out , real dt , int dir , real *flux, GPUstream stream) {
  const int dim = 256;
  if (dir == DIR_X) {
    set_halo_values_x<<<(nz * NUM_VARS + dim - 1) / dim, dim, 0, stream>>>(state_forcing);
    compute_tendencies_x<<<(nz * (nx + 1) + dim - 1) / dim, dim, 0, stream>>>(state_forcing,flux,dt);
    compute_tendencies_x_r<<< (nz * nx * NUM_VARS + dim - 1) / dim, dim, 0, stream>>>(state_forcing,flux,dt, state_init,state_out);
  } else if (dir == DIR_Z) {
    set_halo_values_z<<<(nx+2*hs * NUM_VARS + dim - 1) / dim, dim, 0, stream>>>(state_forcing);
    compute_tendencies_z<<<(nx * (nz + 1) + dim - 1) / dim, dim, 0, stream>>>(state_forcing,flux,dt);
    compute_tendencies_z_r<<< (nx * nz * NUM_VARS + dim - 1) / dim, dim, 0, stream>>>(state_forcing,flux,dt, state_init,state_out);
  }
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
__global__ void compute_tendencies_x( real *state , real *flux , real dt) {
  real stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];
  //Compute the hyperviscosity coefficient
  real hv_coef = -hv_beta * dx / (16*dt);
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int k = id / (nx + 1);
  int i = id % (nx + 1);
  //Compute fluxes in the x-direction for each cell
  if(k < nz) {
    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (int ll=0; ll<NUM_VARS; ll++) {
      for (int s=0; s < sten_size; s++) {
        int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+s;
        stencil[s] = state[inds];
      }
      //Fourth-order-accurate interpolation of the state
      vals[ll] = (-stencil[0] + 7*(stencil[1] + stencil[2]) - stencil[3])/12;
      //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
      d3_vals[ll] = (-stencil[0] + 3*(stencil[1] - stencil[2]) + stencil[3]) * hv_coef;
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    double r = vals[ID_DENS] + hy_dens_cell[k+hs];
    double u = vals[ID_UMOM] / r;
    double w = vals[ID_WMOM] / r;
    double t = ( vals[ID_RHOT] + hy_dens_theta_cell[k+hs] ) / r;
    double p = C0*pow((r*t),gamm);

    //Compute the flux vector
    flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u     - d3_vals[ID_DENS];
    flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*u+p - d3_vals[ID_UMOM];
    flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*w   - d3_vals[ID_WMOM];
    flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*t   - d3_vals[ID_RHOT];
  }
}

__global__ void compute_tendencies_x_r( real *state , real *flux , real dt, real * state_init,real *state_out) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int k = (id / nx) % nz;
  int i = id % nx;
  int ll = id / nx / nz;
  if(ll < NUM_VARS) {
    int indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i  ;
    int indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
    real tend = -( flux[indf2] - flux[indf1] ) / dx;
    int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
    state_out[inds] = state_init[inds] + dt * tend;
  }
}

//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
__global__ void compute_tendencies_z( real *state , real *flux , real dt) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = id % nx;
  int k = id / nx;
  real r,u,w,t,p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coefficient
  hv_coef = -hv_beta * dz / (16*dt);
  //Compute fluxes in the x-direction for each cell
  if(k < nz + 1){
    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (int ll=0; ll<NUM_VARS; ll++) {
      for (int s=0; s<sten_size; s++) {
        int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
        stencil[s] = state[inds];
      }
      //Fourth-order-accurate interpolation of the state
      vals[ll] = (-stencil[0] + 7*(stencil[1] + stencil[2]) - stencil[3])/12;
      //First-order-accurate interpolation of the third spatial derivative of the state
      d3_vals[ll] = (-stencil[0] + 3*(stencil[1] - stencil[2]) + stencil[3]) * hv_coef;
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    r = vals[ID_DENS] + hy_dens_int[k];
    u = vals[ID_UMOM] / r;
    w = vals[ID_WMOM] / r;
    t = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r;
    p = C0*pow((r*t),gamm) - hy_pressure_int[k];
    //Enforce vertical boundary condition and exact mass conservation
    if (k == 0 || k == nz) {
      w                = 0;
      d3_vals[ID_DENS] = 0;
    }

    //Compute the flux vector with hyperviscosity
    flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - d3_vals[ID_DENS];
    flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - d3_vals[ID_UMOM];
    flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - d3_vals[ID_WMOM];
    flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - d3_vals[ID_RHOT];
  }

}

__global__ void compute_tendencies_z_r( real *state , real *flux , real dt, real * state_init,real *state_out) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = id % nx;
  int k = (id / nx) % nz;
  int ll = id / nx / nz;
  if(ll < NUM_VARS){
    int indf1 = ll*(nz+1)*(nx+1) + (k  )*(nx+1) + i;
    int indf2 = ll*(nz+1)*(nx+1) + (k+1)*(nx+1) + i;
    real tend = -( flux[indf2] - flux[indf1] ) / dz;
    if (ll == ID_WMOM) {
      int inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      tend = tend - state[inds]*grav;
    }
    int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
    state_out[inds] = state_init[inds] + dt * tend;
  }
}

//Set this MPI task's halo values in the x-direction. This routine will require MPI
__global__ void set_halo_values_x( real *state ) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int ll = id / nz;
  int k = id % nz;
  if(ll < NUM_VARS){
    state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + 0      ] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs-2];
    state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + 1      ] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs-1];
    state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs  ] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs     ];
    state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs+1] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs+1   ];
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
__global__ void set_halo_values_z( real *state ) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = id % (nx + 2 * hs);
  int ll = id / (nx + 2 * hs);
  if(ll < NUM_VARS){
    if (ll == ID_WMOM) {
      state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = 0.;
      state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = 0.;
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = 0.;
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = 0.;
    } else if (ll == ID_UMOM) {
      state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i] * hy_dens_cell_const[0];
      state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i] * hy_dens_cell_const[1];
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i] * hy_dens_cell_const[2];
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i] * hy_dens_cell_const[3];
    } else {
      state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
      state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
    }
  }
}

void sci_init() {
  int    i, k, ii, kk, ll, inds;
  real x, z, r, u, w, t, hr, ht;

  nranks = 1;
  myrank = 0;
  // i_beg = 0;
  left_rank = 0;
  right_rank = 0;

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  // k_beg = 0;
  mainproc = (myrank == 0);

  //Allocate the model data
  host_state              = (real *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(real) );
  host_state_tmp          = (real *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(real) );
  host_flux               = (real *) malloc( (nx+1)*(nz+1)*NUM_VARS*sizeof(real) );

  host_hy_dens_cell       = (real *) malloc( (nz+2*hs)*sizeof(real) );
  host_hy_dens_theta_cell = (real *) malloc( (nz+2*hs)*sizeof(real) );
  host_hy_dens_int        = (real *) malloc( (nz+1)*sizeof(real) );
  host_hy_dens_theta_int  = (real *) malloc( (nz+1)*sizeof(real) );
  host_hy_pressure_int    = (real *) malloc( (nz+1)*sizeof(real) );

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = dmin(dx,dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  //If I'm the main process in MPI, display some grid information
  if (mainproc) {
    // printf( "nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    // printf( "dx,dz: %lf %lf\n",dx,dz);
    // printf( "dt: %lf\n",dt);
  }

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (k=0; k<nz+2*hs; k++) {
    for (i=0; i<nx+2*hs; i++) {
      //Initialize the state to zero
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        host_state[inds] = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

          //Store into the fluid state array
          inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          host_state[inds] = host_state[inds] + r                         * qweights[ii]*qweights[kk];
          inds = ID_UMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          host_state[inds] = host_state[inds] + (r+hr)*u                  * qweights[ii]*qweights[kk];
          inds = ID_WMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          host_state[inds] = host_state[inds] + (r+hr)*w                  * qweights[ii]*qweights[kk];
          inds = ID_RHOT*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          host_state[inds] = host_state[inds] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        host_state_tmp[inds] = host_state[inds];
      }
    }
  }
  //Compute the hydrostatic background state over vertical cell averages
  for (k=0; k<nz+2*hs; k++) {
    host_hy_dens_cell      [k] = 0.;
    host_hy_dens_theta_cell[k] = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (k_beg + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
      host_hy_dens_cell      [k] = host_hy_dens_cell      [k] + hr    * qweights[kk];
      host_hy_dens_theta_cell[k] = host_hy_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  //Compute the hydrostatic background state at vertical cell interfaces
  for (k=0; k<nz+1; k++) {
    z = (k_beg + k)*dz;
    if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
    host_hy_dens_int      [k] = hr;
    host_hy_dens_theta_int[k] = hr*ht;
    host_hy_pressure_int  [k] = C0*pow((hr*ht),gamm);
  }

  //Copy the hydrostatic background state to the device
  init_device_var();

  GPUMemcpyHtoD((GPUdeviceptr) hy_dens_cell_ptr      , host_hy_dens_cell      , (nz+2*hs)*sizeof(real) );
  GPUMemcpyHtoD((GPUdeviceptr) hy_dens_theta_cell_ptr, host_hy_dens_theta_cell, (nz+2*hs)*sizeof(real) );
  GPUMemcpyHtoD((GPUdeviceptr) hy_dens_int_ptr       , host_hy_dens_int       , (nz+1)*sizeof(real) );
  GPUMemcpyHtoD((GPUdeviceptr) hy_dens_theta_int_ptr , host_hy_dens_theta_int , (nz+1)*sizeof(real) );
  GPUMemcpyHtoD((GPUdeviceptr) hy_pressure_int_ptr   , host_hy_pressure_int   , (nz+1)*sizeof(real) );

  GPUMemAlloc((GPUdeviceptr *) (void**) &state     , (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(real) );
  GPUMemAlloc((GPUdeviceptr *) (void**) &state_tmp , (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(real) );
  GPUMemAlloc((GPUdeviceptr *) (void**) &flux      , (nx+1)*(nz+1)*NUM_VARS*sizeof(real) );

  GPUMemcpyHtoD((GPUdeviceptr) state    , host_state    , (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(real) );
  GPUMemcpyHtoD((GPUdeviceptr) state_tmp, host_state_tmp, (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(real) );
  GPUMemcpyHtoD((GPUdeviceptr) flux     , host_flux     , (nx+1)*(nz+1)*NUM_VARS*sizeof(real)      );

  real temp[4];
  temp[0] = host_hy_dens_cell[0] / host_hy_dens_cell[hs];
  temp[1] = host_hy_dens_cell[1] / host_hy_dens_cell[hs];
  temp[2] = host_hy_dens_cell[nz + hs] / host_hy_dens_cell[nz + hs - 1];
  temp[3] = host_hy_dens_cell[nz + hs + 1] / host_hy_dens_cell[nz + hs - 1];
  GPUMemcpyHtoD((GPUdeviceptr)hy_dens_cell_const_ptr, temp, 4*sizeof(real) );
}


//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void injection( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
}


//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void density_current( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.);
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void gravity_waves( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_bvfreq(z,0.02,hr,ht);
  r = 0.;
  t = 0.;
  u = 15.;
  w = 0.;
}


//Rising thermal
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void thermal( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.);
}


//Colliding thermals
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void collision( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t = t + sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}


//Establish hydrostatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_theta( real z , real &r , real &t ) {
  const real theta0 = 300.;  //Background potential temperature
  const real exner0 = 1.;    //Surface-level Exner pressure
  real       p,exner,rt;
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}


//Establish hydrostatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_bvfreq( real z , real bv_freq0 , real &r , real &t ) {
  const real theta0 = 300.;  //Background potential temperature
  const real exner0 = 1.;    //Surface-level Exner pressure
  real       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad ) {
  real dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.) {
    return amp * pow(cos(dist),2.);
  } else {
    return 0.;
  }
}

void finalize() {
  GPUMemFree((GPUdeviceptr) state );
  GPUMemFree((GPUdeviceptr) state_tmp );
  GPUMemFree((GPUdeviceptr) flux );
  GPUMemFree((GPUdeviceptr) hy_dens_cell_ptr );
  GPUMemFree((GPUdeviceptr) hy_dens_theta_cell_ptr );
  GPUMemFree((GPUdeviceptr) hy_dens_int_ptr );
  GPUMemFree((GPUdeviceptr) hy_dens_theta_int_ptr );
  GPUMemFree((GPUdeviceptr) hy_pressure_int_ptr );

  free( host_state );
  free( host_state_tmp );
  free( host_flux );
  free( host_hy_dens_cell );
  free( host_hy_dens_theta_cell );
  free( host_hy_dens_int );
  free( host_hy_dens_theta_int );
  free( host_hy_pressure_int );
}


//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
void reductions(double &mass , double &te) {
  GPUMemcpyDtoH( host_state , (GPUdeviceptr)state , (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(real) );
  GPUMemcpyDtoH( host_state_tmp , (GPUdeviceptr) state_tmp , (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(real) );
  GPUMemcpyDtoH( host_flux , (GPUdeviceptr) flux , (nx+1)*(nz+1)*NUM_VARS*sizeof(real) );

  mass = 0;
  te   = 0;
  for (int k=0; k<nz; k++) {
    for (int i=0; i<nx; i++) {
      int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      double r  =   host_state[ind_r] + host_hy_dens_cell[hs+k];             // Density
      double u  =   host_state[ind_u] / r;                              // U-wind
      double w  =   host_state[ind_w] / r;                              // W-wind
      double th = ( host_state[ind_t] + host_hy_dens_theta_cell[hs+k] ) / r; // Potential Temperature (theta)
      double p  = C0*pow(r*th,gamm);                               // Pressure
      double t  = th / pow(p0/p,rd/cp);                            // Temperature
      double ke = r*(u*u+w*w);                                     // Kinetic Energy
      double ie = r*cv*t;                                          // Internal Energy
      mass += r        *dx*dz; // Accumulate domain mass
      te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    }
  }
}

} // namespace SciComputeRaw