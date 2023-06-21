/*=========================================================================
 * This is a MEX-file for MATLAB.
 * Author: Scott Albert
 * Email: salbert8@jhu.edu
 * Institution: Johns Hopkins University
 * Lab: Laboratory for Computational Motor Control
 * Advisor: Reza Shadmehr
 * Date: July 25th, 2017
 * Version: 1.1
 *
 * Summary: This function computes the value of the expected complete
 * log-likelihood function. It will need to be built using a C/C++ compiler
 * in order to use it for generarlized expectation maximization.
 * 
 * Notes: For more information about this package see README.pdf.
 *        To compile this function go to the MATLAB command line and enter:
 *              mex expected_complete_log_likelihood_mex.cpp
 *
 * Input description:
 *    parameters: the current estimate of the two-state model parameters
 *    y: the motor output on each trial
 *    e: the error experienced by the subject on each trial
 *    c: a model parameter that is assumed invariant
 *    xnN: This is shorthand for the quantity x(n|N). It is the smoothed
 *        Kalman state expectation  E[x(n)|y(1),y(2),...,y(N)].
 *    VnN: This is shorthand for the quantity V(n|N). It is the smoothed
 *        Kalman state variance  var(x(n)|y(1),y(2),...,y(N)).
 *    Vnp1nN: This is shorthand for the quantity V(n+1,n|N). It is the
 *        smoothed Kalman covariance of consecutive states, also written as
 *        cov(x(n+1),x(n)|y(1),y(2),...,y(N)).
 *  
 * Output description:
 *     likelihood: the expected complete log-likelihood
 *
 *=======================================================================*/

#include "mex.h"
#include <math.h>
#include <string.h>
#define PI 3.141592653589793

// local function declarations
double trace(double *);
void matrix22inverse(double *, double *);
double matrix22determinant(double *); 
void matrix22transpose(double *, double *);
double * getMatrixFromCell(mxArray *, mxArray *, mwIndex);
void matrix22matrix22mult(double *, double *, double *);
void matrix22vector21mult(double *, double *, double *);
void vector21matrix22mult(double *, double *, double *);
void vector21vector12mult(double *, double *, double *);
void matrix22matrix22sum(double *, double *, double *);
double vector12matrix22vector21mult(double *, double *, double *);
void printMatrix22(double *);
void printVector21(double *);
void printScalar(double);
double dot(double *, double *);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ///////////////////////////////////////////////////////////////////////
    /////////////////// declare and load inputs ///////////////////////////
    ///////////////////////////////////////////////////////////////////////
    // parameters that specify the paradigm and behavior
    double *parameters, *y, *e, *c;
    // the two state model parameters
    double aS, aF, bS, bF, sigmax2, sigmau2, xS1, xF1, sigma12;
    // parameters obtained from the Kalman Smoother
    mxArray *VnNCellArray, *VnNCell, *Vnp1nNCellArray, *Vnp1nNCell;
    mxArray *xnNCellArray, *xnNCell;
    //a variable to hold trial numbers
    mwIndex n;
    //declares matrix dimensions (number of trials)
    size_t N;
    //declares matrix dimenions (number of states)
    size_t nStates = (size_t) 2;
    
    //gets pointers for Kalman vectors and matrices
    double *xnN; // this is E[x(n)|y(1),y(2),...,y(N)]
    double *xnp1N; // this is E[x(n+1)|y(1),y(2),...,y(N)]
    double *VnN; // this is var(x(n)|y(1),y(2),...,y(N))
    double *Vnp1N; // this is var(x(n+1)|y(1),y(2),...,y(N))
    double *Vnp1nN; // this is cov(x(n+1),x(n)|y(1),y(2),...,y(N))
     // this is cov(x(n),x(n+1)|y(1),y(2),...,y(N))
    double *Vnnp1N = (double*) mxMalloc(4 * sizeof(double));
    
    //gets the inputs to the mex function
    parameters = mxGetPr(prhs[0]);
    y = mxGetPr(prhs[1]); //the motor output
    e = mxGetPr(prhs[2]); //the error
    c = mxGetPr(prhs[3]); //the c parameter
    xnNCellArray = const_cast<mxArray*>(prhs[4]);
    VnNCellArray = const_cast<mxArray*>(prhs[5]);
    Vnp1nNCellArray = const_cast<mxArray_tag*>(prhs[6]);
    
    //assigns two state model parameters
    aS = parameters[0];
    aF = parameters[1];
    bS = parameters[2];
    bF = parameters[3];
    xS1 = parameters[4];
    xF1 = parameters[5];     
    sigmax2 = parameters[6];
    sigmau2 = parameters[7];
    sigma12 = parameters[8];
    
    //stores the number of trials
    //gets the dimensions for either a column or row vector input
    if(mxGetM(prhs[1]) > mxGetN(prhs[1])) { 
        N = mxGetM(prhs[1]);
    }
    else if(mxGetN(prhs[1]) > mxGetM(prhs[1])) { 
        N = mxGetN(prhs[1]);
    }
    
    //constructs the V1 matrix (variance of the initial state)
    double *V1 = (double*) mxMalloc(4 * sizeof(double));
    V1[0] = sigma12;
    V1[1] = 0;
    V1[2] = 0;
    V1[3] = sigma12;
    
    //computes V1inv
    double *V1inv = (double*) mxMalloc(4 * sizeof(double));
    matrix22inverse(V1, V1inv);
    
    //constructs the x1 vector
    double *x1 = (double*) mxMalloc(2 * sizeof(double));
    x1[0] = xS1;
    x1[1] = xF1;
    
    ///////////////////////////////////////////////////////////////////////
    //////////////////////// specify A, b, and Q  /////////////////////////
    ///////////////////////////////////////////////////////////////////////
    double *A = (double*) mxMalloc(4 * sizeof(double));
    A[0] = aS;
    A[1] = 0;
    A[2] = 0;
    A[3] = aF;
    
    double *b = (double*) mxMalloc(2 * sizeof(double));
    b[0] = bS;
    b[1] = bF;
    
    double *Q = (double*) mxMalloc(4 * sizeof(double));
    Q[0] = sigmax2;
    Q[1] = 0;
    Q[2] = 0;
    Q[3] = sigmax2;
    
    ///////////////////////////////////////////////////////////////////////
    ////////// Computes all terms in the likelihood function that /////////
    ////////// depend on A, Q, and b //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    double *Qinv = (double*) mxMalloc(4 * sizeof(double));
    matrix22inverse(Q,Qinv);

    double *QinvA = (double*) mxMalloc(4 * sizeof(double));
    matrix22matrix22mult(Qinv,A,QinvA);    
    
    double *Qinvb = (double*) mxMalloc(2 * sizeof(double));
    matrix22vector21mult(Qinv,b,Qinvb);
    
    double *At = (double*) mxMalloc(4 * sizeof(double));
    matrix22transpose(A,At);
    
    double *AtQinv = (double*) mxMalloc(4 * sizeof(double));
    matrix22matrix22mult(At,Qinv,AtQinv);
    
    double *AtQinvA = (double*) mxMalloc(4 * sizeof(double));
    matrix22matrix22mult(AtQinv,A,AtQinvA);
    
    double *AtQinvb = (double*) mxMalloc(2 * sizeof(double));
    matrix22vector21mult(AtQinv,b,AtQinvb);
    
    double *bt = (double*) mxMalloc(2 * sizeof(double));
    bt[0] = b[0]; bt[1] = b[1];
    
    double *btQinv = (double*) mxMalloc(2 * sizeof(double));
    vector21matrix22mult(bt,Qinv,btQinv);   
    
    double *btQinvA = (double*) mxMalloc(2 * sizeof(double));
    vector21matrix22mult(btQinv,A,btQinvA);
    
    double btQinvb = dot(btQinv,b);
    
    double logdetQ = log(matrix22determinant(Q));
    
    ///////////////////////////////////////////////////////////////////////
    ////////////// computes the expected complete log-likelihood //////////
    ///////////////////////////////////////////////////////////////////////
    
    //declares double terms for the calculation of the likelihood function
    // in five different parts
    double term1 = 0;
    double term2 = 0;
    double term3 = 0;
    double term4 = 0;
    double term5 = 0;
    
    //TERM 1: one of the parts of the expected complete log-likelihood
    //function derived from the likelihood of observing the motor output
    //given the states
    double *VnNplusxnNxnNt = (double*) mxMalloc(4 * sizeof(double));
    double *xnNxnNt = (double*) mxMalloc(4 * sizeof(double));
    for(n = 0; n < N; n++) { 
        //gets the current xnN and VnN
        xnN = getMatrixFromCell(xnNCellArray, xnNCell, n);
        VnN = getMatrixFromCell(VnNCellArray, VnNCell, n);
        
        //computes xnN * xnN(transpose)
        vector21vector12mult(xnN, xnN, xnNxnNt);
        
        //computes the sum of xnN * xnN(transpose) + VnN
        matrix22matrix22sum(VnN, xnNxnNt, VnNplusxnNxnNt);
        
        // adds y(n) * y(n) to term1
        term1 += y[n]*y[n];
        
        // adds c(transpose) * (VnN + xnN*xnN(transpose) * c to term1
        term1 += vector12matrix22vector21mult(c, VnNplusxnNxnNt, c);
        
        // subtracts -2 * y(n) * c(transpose) * xnN
        term1 += -2.0*y[n]*dot(c, xnN);
    }
    //completes the computation of term1 by dividing by -sigmau2
    term1 = -term1 / (2.0*sigmau2);
    
    //TERM 2: one of the parts of the expected complete log-likelihood
    //function derived from the likelihood of observing the state on trial
    //n+1 given the state on trial n
    double *QinvVnp1N = (double*) mxMalloc(4 * sizeof(double));
    double *QinvAVnnp1N = (double*) mxMalloc(4 * sizeof(double));
    double *AtQinvVnp1nN = (double*) mxMalloc(4 * sizeof(double));
    double *AtQinvAVnN = (double*) mxMalloc(4 * sizeof(double));
    
    //the main loop of the function where term2 is computed
    for(n = 0; n < (N - 1); n++) {
        
        // get the current xnN, VnN, and Vnp1nN
        xnN = getMatrixFromCell(xnNCellArray, xnNCell, n);
        xnp1N = getMatrixFromCell(xnNCellArray, xnNCell, n+1);
        VnN = getMatrixFromCell(VnNCellArray, VnNCell, n);
        Vnp1N = getMatrixFromCell(VnNCellArray, VnNCell, n+1);
        Vnp1nN = getMatrixFromCell(Vnp1nNCellArray, Vnp1nNCell, n);
        matrix22transpose(Vnp1nN, Vnnp1N); //computes the transpose of Vnp1nN
        
        //increments term2 by xnp1N(transpose) * Qinv(n) * xnp1N
        term2 += vector12matrix22vector21mult(xnp1N, Qinv, xnp1N);
        
        //computes Qinv(n) * Vnp1N
        matrix22matrix22mult(Qinv, Vnp1N, QinvVnp1N);
        
        //increments term2 by trace(Qinv(n) * Vnp1N)
        term2 += trace(QinvVnp1N);
        
        //increments term2 by -xnp1N(transpose) * Qinv * A * xnN
        term2 += -vector12matrix22vector21mult(xnp1N, QinvA, xnN);
        
        //computes QinvAVnnp1N
        matrix22matrix22mult(QinvA, Vnnp1N, QinvAVnnp1N);

        //increments term2 by -trace(Qinv * A * Vnnp1N)
        term2 += -trace(QinvAVnnp1N);
        
        //increments term2 by -xnp1N(transpose) * Qinv * b * r
        term2 += -dot(xnp1N, Qinvb)*e[n];
             
        //increments term2 by -xnN(transpose) * A(transpose) * Qinv * xnp1N
        term2 += -vector12matrix22vector21mult(xnN, AtQinv, xnp1N);
             
        //computes At * Qinv * Vnp1nN
        matrix22matrix22mult(AtQinv, Vnp1nN, AtQinvVnp1nN);
        
        //increments term2 by -trace(At * Qinv * Vnp1nN)
        term2 += -trace(AtQinvVnp1nN);
        
        //increments term2 by xnN(transpose) * A(transpose) * Qinv * A * xnN
        term2 += vector12matrix22vector21mult(xnN, AtQinvA, xnN);
        
        //computes At * Qinv * A * VnN
        matrix22matrix22mult(AtQinvA, VnN, AtQinvAVnN);
        
        //increments term2 by trace(At * Qinv * A * VnN)
        term2 += trace(AtQinvAVnN);
        
        //increments term2 by xnN(transpose) * A(transpose) * Qinv * b * r
        term2 += dot(xnN, AtQinvb)*e[n];
        
        //increments term2 by -r(n) * b(transpose) * Qinv * xnp1N
        term2 += -e[n]*dot(btQinv, xnp1N);
        
        //increments term2 by r(n) * b(transpose) * Qinv * A * xnN
        term2 += e[n]*dot(btQinvA, xnN);
        
        //increments term2 by r(n) * b(transpose) * Qinv * b * r(n)
        term2 += e[n]*btQinvb*e[n];   
    }
    //completes the computation of term2 by dividing by -2
    term2 = -term2/2.0;
    
    //TERM 3: one of the parts of the expected complete log-likelihood
    //function derived from the likelihood of observing the initial state
    double *x1N;
    x1N = getMatrixFromCell(xnNCellArray, xnNCell, 0);
    double *V1N;
    V1N = getMatrixFromCell(VnNCellArray, VnNCell, 0);
    //computes V1inv * V1N
    double *V1invV1N = (double*) mxMalloc(4 * sizeof(double));
    matrix22matrix22mult(V1inv, V1N, V1invV1N);
    term3 += vector12matrix22vector21mult(x1N, V1inv, x1N);
    term3 += trace(V1invV1N);
    term3 += -vector12matrix22vector21mult(x1N, V1inv, x1);
    term3 += -vector12matrix22vector21mult(x1, V1inv, x1N);
    term3 += vector12matrix22vector21mult(x1, V1inv, x1);
    term3 = -term3/2.0;
    
    //TERM 4: one of the parts of the expected complete log-likelihood function
    //derived from the pre-exponential factors
    term4 += -log(matrix22determinant(V1))/2.0 - N*log(sigmau2)/2.0 - 
            3.0*N*log(2.0*PI)/2.0;      
    
    //TERM 5: one of the parts of the expected complete log-likelihood function
    //derived from the pre-exponential factors of x(n+1) given x(n)
    term5 = -logdetQ*(N-1)/2.0;
    
    // create output and assign it the likelihood value
    plhs[0] = mxCreateDoubleScalar(mxREAL);
    double *likelihood = mxGetPr(plhs[0]);
    *likelihood = term1 + term2 + term3 + term4 + term5;
    
    //free memory
    mxFree(Vnnp1N);
    mxFree(V1);
    mxFree(V1inv);
    mxFree(x1);
    mxFree(A);
    mxFree(b);
    mxFree(Q);
    mxFree(Qinv);
    mxFree(QinvA);
    mxFree(Qinvb);
    mxFree(At);
    mxFree(AtQinv);
    mxFree(AtQinvA);
    mxFree(AtQinvb);
    mxFree(bt);
    mxFree(btQinv);
    mxFree(btQinvA);
    mxFree(VnNplusxnNxnNt);
    mxFree(xnNxnNt);
    mxFree(QinvVnp1N);
    mxFree(QinvAVnnp1N);
    mxFree(AtQinvVnp1nN);
    mxFree(AtQinvAVnN);
    mxFree(V1invV1N);
}


//////////////////////////////////////////////////////////////////////////
///////////////////////// Local functions ////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// a function that computes the trace of a 2 x 2 matrix
double trace(double *matrix) {
    double output = matrix[0] + matrix[3];
    return output;
}

// a function that computes the inverse of a matrix
void matrix22inverse(double *matrix, double *inverse) {
    // computes the determinant of the matrix
    double determinant = matrix[0]*matrix[3] - matrix[1]*matrix[2];
    
    // constructs the inverse matrix
    inverse[0] = matrix[3] / determinant;
    inverse[1] = -matrix[1] / determinant;
    inverse[2] = -matrix[2] / determinant;
    inverse[3] = matrix[0] / determinant;
}

double matrix22determinant(double *matrix) {
    double determinant = matrix[0]*matrix[3] - matrix[1]*matrix[2];
    return determinant;
}

// a function that returns the transpose of a matrix
void matrix22transpose(double *matrix, double *transpose) {
    transpose[0] = matrix[0];
    transpose[3] = matrix[3];
    transpose[1] = matrix[2];
    transpose[2] = matrix[1];
}

// a function that returns a 2 x 2 matrix that is stored within a cell array
double * getMatrixFromCell(mxArray *cellArray, mxArray *cell, mwIndex elementNumber) {
    //gets a pointer to the 1st cell
    cell = mxGetCell(cellArray,elementNumber);
    double *cellPointer = mxGetPr(cell);
    return cellPointer;
}

// multiplies 2, 2 x 2 matrices M1 and M2 and stores the result in M3
void matrix22matrix22mult(double *M1, double *M2, double *M3) { 
    M3[0] = M1[0]*M2[0] + M1[2]*M2[1];
    M3[1] = M1[1]*M2[0] + M1[3]*M2[1];
    M3[2] = M1[0]*M2[2] + M1[2]*M2[3];
    M3[3] = M1[1]*M2[2] + M1[3]*M2[3];
}

// multiplies a 2 x 2 matrix by a 2 x 1 vector
void matrix22vector21mult(double *M, double *v, double *output) {
    output[0] = M[0]*v[0] + M[2]*v[1];
    output[1] = M[1]*v[0] + M[3]*v[1];
}

// multiplies a 1 x 2 vector by a 2 x 2 matrix
void vector21matrix22mult(double *v, double *M, double *output) {
    output[0] = v[0]*M[0] + v[1]*M[1];
    output[1] = v[0]*M[2] + v[1]*M[3];
}

// multiplies a 2 x 1 vector by a 1 x 2 vector
void vector21vector12mult(double *v1, double *v2, double *output) {
    output[0] = v1[0]*v2[0];
    output[1] = v1[1]*v2[0];
    output[2] = v1[0]*v2[1];
    output[3] = v1[1]*v2[1];
}

// multiplies a 1 x 2 vector by a 2 x 2 matrix by a 2 x 1 vector
double vector12matrix22vector21mult(double *v, double *M, double *x) {
    double output = v[0]*x[0]*M[0] + v[1]*x[0]*M[1] + v[0]*x[1]*M[2] + v[1]*x[1]*M[3];
    return output;
}

// adds 2, 2 x 2 matrices
void matrix22matrix22sum(double *M1, double *M2, double *output) {
    output[0] = M1[0] + M2[0];
    output[1] = M1[1] + M2[1];
    output[2] = M1[2] + M2[2];
    output[3] = M1[3] + M2[3];
}

// performs the dot product on 2, 2 x 1 vectors
double dot(double *v1, double *v2) {
    double result = v1[0]*v2[0] + v1[1]*v2[1];
    return result;
}

// prints out a 2 x 2 matrix
void printMatrix22(double *matrix) {
    mexPrintf("%g %g \n %g %g \n",matrix[0],matrix[2],matrix[1],matrix[3]);
}

// prints our a 2 x 1 vector
void printVector21(double *vector) {
    mexPrintf("%g \n %g \n",vector[0],vector[1]);
}

// prints a scalar value
void printScalar(double value) {
    mexPrintf("%f \n", value);
}
