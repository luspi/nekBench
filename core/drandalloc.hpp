static dfloat* drandAlloc(int N)
{
  dfloat* v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n = 0; n < N; ++n)
    v[n] = drand48();

  return v;
}
