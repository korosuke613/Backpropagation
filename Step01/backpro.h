/*
 *   backpro.h
 */

#ifndef __BACKPRO_H
#define __BACKPRO_H

typedef struct {
  int n1;  /* number of input units */
  int n2;  /* number of units in the second layer */
  double *x; /* input units */
  double *u; /* output of the second layer units */
  double z;  /* output unit */
  double **s; /* s[j][k]: connection weights */
  double *w; /* w[j]: connection weights to the ouput unit */
} BPNN;

#define RAND_SEED 20180509

BPNN* bpnn_new(int n1, int n2);
void free_bpnn(BPNN* this);

#endif /* BACKPRO_H*/
