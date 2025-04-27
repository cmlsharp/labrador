#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <gmp.h>

#include "polx.h"
#include "polz.h"
#include "fips202.h"
#include "malloc.h"
#include "labrador.h"
#include "chihuahua.h"
#include "pack.h"

// 2**64+1
const char *MOD_STR = "18446744073709551617";

// number of repetitions (l in paper);
#define ELL 8
// Number of repititons needed for Z_q constraints (128 bits of security)
#define ELL2 ((LOGQ+128-1)/LOGQ)


#define R1CSVECS (4+ELL)
#define EVALVECS 2
#define NWITVECS (R1CSVECS + EVALVECS)

typedef struct {
    size_t k; // R1CS matrix rows
    size_t n; // R1CS matrix columns
    size_t m[3]; // Commitment sizes for each round
    int64_t gnorm; // Norm bound on coefficients of g_is
    uint64_t g_bw; // Max bit width of g_is
    int64_t vnorm; // Norm bound on coefficients of quotients
    uint64_t v_bw; // Max bit width of quotients
    int64_t hnorm; // Norm bound on carries
    uint64_t h_bw; // Max bit width of carries
    int64_t G[N]; // Value guaranteed to make ith carry equation non-negative
    int64_t ac[N+1]; // Aux constants that "correct" for having added G[N] in carry equation
    int64_t carry_eqn_bound; // Largest value that could possibly arise from any carry equation
    uint64_t eval_vec_len;
} R1CSParams;

uint64_t ceildiv(uint64_t a, uint64_t b)
{
    return (a+b-1)/b;
}

uint64_t significant_bits(uint64_t x)
{
    if (x == 0) return 0;
    return 64 - __builtin_clzll(x);
}

int64_t zz_toint64(zz const *a)
{
    assert(LOGQ < 64);
    assert(L * 14 < 64);
    int64_t r = a->limbs[L-1];
    for (size_t i = 1; i != L; i++) {
        r <<= 14;
        r += a->limbs[L-1-i];
    }
    return r;

}

uint64_t beta_squared(R1CSParams const *rp)
{
    return ceil(128. / 30. * (((3+ELL)*rp->k+rp->n)*(N/2. + 3) + 2*ELL*(N*rp->g_bw + rp->v_bw + (N-1)*rp->h_bw)));
}

void new_r1cs_params(R1CSParams *rp, size_t k, size_t n, size_t m[3])
{
    rp->k = k;
    rp->n = n;
    memcpy(rp->m, m, 3*sizeof *m);
    rp->gnorm = ((4 + ELL)*k+n)*(N/2+3);
    rp->vnorm = 2*rp->gnorm;
    rp->hnorm = (rp->gnorm + 2*rp->vnorm)/2;

    rp->g_bw = significant_bits(rp->gnorm) + 1;
    rp->v_bw = significant_bits(rp->vnorm) + 1;
    rp->h_bw = significant_bits(rp->hnorm);

    // gpmaxword
    for (size_t i = 0; i != N; i++) {
        rp->G[i] = rp->gnorm;
    }
    rp->G[0] += rp->vnorm;
    rp->G[N-1] += 2*rp->vnorm;

    rp->ac[N] = 0;
    // Aux consts
    for (size_t i = 0; i != N; i++) {
        rp->ac[N] += rp->G[i];
        rp->ac[i] = rp->ac[N] % 2;
        rp->ac[N] >>= 1;
    }

    // Sanity check, just to make sure q is big enough that the carry equations don't overflow
    rp->carry_eqn_bound = 0;
    for (size_t i = 0; i != N; i++) {
        int64_t lhs_min = 0, lhs_max = 0;
        int64_t rhs_min = 0, rhs_max = 0;
        lhs_min += rp->G[i] - rp->gnorm;
        // could probably be 2*gnorm instead which is tighter since g is constrained to be "correct" by other constraints
        lhs_max += rp->G[i] - rp->gnorm + (1l << rp->g_bw);
        rhs_min += rp->ac[i];
        rhs_max += rp->ac[i];
        if (i == 0) {
            rhs_min -= rp->vnorm;
            rhs_max += (1l << rp->v_bw) - rp->vnorm;
            rhs_max += 2*(1l << rp->h_bw);
        } else if (i == N-1) {
            rhs_min -= 2*rp->vnorm;
            rhs_max += 2*((1l << rp->v_bw) - rp->vnorm);
            lhs_max += 1l << rp->h_bw;
            rhs_min += 2*rp->ac[N];
            rhs_max += 2*rp->ac[N];
        } else {
            lhs_max += 1l << rp->h_bw;
            rhs_max += 2*(1l << rp->h_bw);
        }
        int64_t cur_eqn_bound = MAX(lhs_max - rhs_min, rhs_max - lhs_min);
        rp->carry_eqn_bound = MAX(cur_eqn_bound, rp->carry_eqn_bound);
    }

    int64_t q = zz_toint64(&modulus.q);
    assert(rp->carry_eqn_bound < q);
    rp->eval_vec_len = ELL*rp->g_bw + ceildiv(ELL*rp->v_bw, N) + ceildiv(ELL*(N-1)*rp->h_bw, N);

    uint64_t betasq = beta_squared(rp);
    assert(betasq < (uint64_t) q);
    assert(sqrt((rp->n + (3+ELL)*rp->k)*(N/2. + 3)*betasq) + betasq/2. < (double) q);
}


int64_t polz_getcoeff_int64(polz const *a, int k)
{
    zz coeffzz;
    polz_getcoeff(&coeffzz, a, k);
    return zz_toint64(&coeffzz);
}


void polz_print2(const polz *a)
{
    for (size_t i = 0; i != N; i++) {
        int64_t x = a->limbs[0].c[i];
        for (size_t j = 1; j != L; j++) {
            x += a->limbs[j].c[i] * 1l << (14*j);
        }
        printf("%5ldx^%zu", x,i);
        if (i != N-1) {
            printf(" + ");
        }
    }
    printf("\n");
}

void polx_print2(const polx *a)
{
    polz z;
    polz_frompolx(&z, a);
    polz_center(&z);
    polz_print2(&z);
}


mpz_t *new_mpz_array(size_t len)
{
    mpz_t *arr = _malloc(len * sizeof *arr);
    for (size_t i = 0; i != len; i++) {
        mpz_init(arr[i]);
    }
    return arr;
}

void free_mpz_array(mpz_t *arr, size_t len)
{
    for (size_t i = 0; i != len; i++) {
        mpz_clear(arr[i]);
    }
    free(arr);
}

void poly_print(const poly *a, const char *fmt, ...)
{
    size_t i;

    va_list args;
    va_start(args, fmt);
    //vprintf(fmt, args);
    //printf(" = np.array([");
    for(i=0; i<N; i++) {
        //printf("%2zu: ",i);
        printf("%d, ",a->vec->c[i]);
        //printf("\n");
    }
    //printf("], dtype='object')\n");
    printf("\n");
    va_end(args);
}


// update h with hash of constraints in sparsecnst
// TODO update w/ sparse representation
void sparsecnst_hash(uint8_t h[16], sparsecnst const *cnst, size_t const *n)
{
    size_t deg = MAX(cnst->deg,1);
    polz t[deg];
    __attribute__((aligned(16)))
    uint8_t hashbuf[deg*N*QBYTES];
    shake128incctx shakectx;

    shake128_inc_init(&shakectx);
    shake128_inc_absorb(&shakectx, h, 16);

    polzvec_frompolxvec(t, cnst->b, deg);
    polzvec_bitpack(hashbuf, t, deg);
    shake128_inc_absorb(&shakectx, hashbuf, deg*N*QBYTES);

    // TODO: SPARSE REPRESENTATION FIX
    for (size_t j = 0; j != cnst->nz; j++) {
        for (size_t k = 0; k != n[cnst->idx[j]]; k++) {
            polzvec_frompolxvec(t, cnst->phi[j] + k*deg, deg);
            polzvec_bitpack(hashbuf, t, deg);
            shake128_inc_absorb(&shakectx, hashbuf, deg*N*QBYTES);
        }
    }

    if (cnst->a->coeffs) {
        // Should I also hash rows and cols? Probably should to be safe
        for (size_t i = 0; i != cnst->a->len; i++) {
            // TODO eventually generalize this?
            polzvec_frompolxvec(t, cnst->a->coeffs + i, 1);
            polzvec_bitpack(hashbuf, t, 1);
            shake128_inc_absorb(&shakectx, hashbuf, N*QBYTES);
        }
    }



    shake128_inc_finalize(&shakectx);
    shake128_inc_squeeze(h, 16, &shakectx);
}


// quick and dirty for debugg purposes
void polz_eval(mpz_t output, const polz *t, int64_t coord, mpz_t const *mod)
{
    mpz_t power, coeff, temp;
    mpz_inits(power, coeff, temp, NULL);
    //mpz_set_str(mod, MOD_STR, 10);
    //mpz_set_ui(mod, Q);
    mpz_set_ui(output, 0);
    mpz_set_ui(power, 1);

    for (size_t i = 0; i != N; i++) {
        mpz_set_ui(coeff, 0);

        for (size_t j = 0; j != L; j++) {
            mpz_ui_pow_ui(temp, 2, 14*j);
            mpz_mul_si(temp, temp, t->limbs[j].c[i]);
            mpz_add(coeff, coeff, temp);
            if (mod) mpz_mod(coeff, coeff, *mod);
        }

        mpz_addmul(output, coeff, power);
        if (mod) mpz_mod(output, output, *mod);
        mpz_mul_ui(power, power, coord);
        if (mod) mpz_mod(power, power, *mod);
    }
    mpz_clears(power, coeff, temp, NULL);
}

void poly_eval(mpz_t output, const poly *t, int64_t coord, mpz_t const *mod)
{
    polz z;
    polz_frompoly(&z, t);
    polz_center(&z);
    polz_eval(output, &z, coord, mod);
}


void polx_eval(mpz_t output, polx const *t, int64_t coord, mpz_t const *mod)
{
    polz z;
    polz_frompolx(&z, t);
    polz_center(&z);
    polz_eval(output, &z, coord, mod);

}
// debugging functions
void print_mpz_array(mpz_t const *arr, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        gmp_printf("%Zd ", arr[i]);
    }

    printf("\n");
}

void print_int64_array(int64_t const *arr, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        printf("%" PRId64 " ", arr[i]);
    }

    printf("\n");
}

typedef struct {
    size_t len;
    size_t cap;
    size_t *rows;
    size_t *cols;
    mpz_t *entries;
} mpz_sparsemat;

void init_mpz_sparsemat(mpz_sparsemat *mat, size_t cap)
{
    mat->len = 0;
    mat->cap = cap;
    mat->rows = _malloc(cap * sizeof *mat->rows);
    mat->cols = _malloc(cap * sizeof *mat->cols);
    mat->entries = _malloc(cap * sizeof *mat->entries);
}

void free_mpz_sparsemat(mpz_sparsemat *mat)
{
    for (size_t i = 0; i != mat->len; i++) {
        mpz_clear(mat->entries[i]);
    }
    free(mat->rows);
    free(mat->cols);
    free(mat->entries);
}

void add_entry(mpz_sparsemat *mat, size_t row, size_t col, mpz_t const val)
{
    assert(mat->len < mat->cap);
    mat->rows[mat->len] = row;
    mat->cols[mat->len] = col;
    mpz_init_set(mat->entries[mat->len], val);
    mat->len++;
}

void add_entry_ui(mpz_sparsemat *mat, size_t row, size_t col, unsigned long val)
{
    assert(mat->len < mat->cap);
    mat->rows[mat->len] = row;
    mat->cols[mat->len] = col;
    mpz_init_set_ui(mat->entries[mat->len], val);
    mat->len++;
}

// result += matrix * vector (mod mod)
void sparsematmul_add(mpz_t *result, mpz_sparsemat const *mat, mpz_t const *vector, mpz_t const mod)
{
    for (size_t i = 0; i < mat->len; i++) {
        size_t row = mat->rows[i];
        size_t col = mat->cols[i];
        mpz_addmul(result[row], mat->entries[i], vector[col]);
        mpz_mod(result[row], result[row], mod);
    }
}

// result = matrix * vector (mod mod)
void sparsematmul(mpz_t *result, mpz_sparsemat const *mat, mpz_t const *vector, size_t nrows, mpz_t const mod)
{
    for (size_t i = 0; i != nrows; i++) {
        mpz_set_ui(result[i], 0);
    }
    sparsematmul_add(result, mat, vector, mod);
}


// result += matrix^T*vector (mod mod)
void sparsematmul_trans_add(mpz_t *result, mpz_sparsemat const *mat, mpz_t const *vector, mpz_t const mod)
{
    for (size_t i = 0; i < mat->len; i++) {
        size_t row = mat->rows[i];
        size_t col = mat->cols[i];
        mpz_addmul(result[col], mat->entries[i], vector[row]);
        mpz_mod(result[col], result[col], mod);
    }
}

// result = matrix^T * vector (mod mod)
void sparsematmul_trans(mpz_t *result, mpz_sparsemat const *mat, mpz_t const *vector, size_t ncols, mpz_t const mod)
{
    for (size_t i = 0; i != ncols; i++) {
        mpz_set_ui(result[i], 0);
    }
    sparsematmul_trans_add(result, mat, vector, mod);
}


// matrix represented as 1d array where (i,j)th cell is (matrix + i * cols + j)
// result += matrix * vector
void polx_matmul_add(polx *result, polx const *matrix, polx const *vector, size_t rows, size_t cols)
{
    for (size_t i = 0; i != rows; i++) {
        for (size_t j = 0; j != cols; j++) {
            polx_mul_add(result + i, matrix + i*cols + j, vector + j);
        }
    }
}

// result = matrix * vector
void polx_matmul(polx *result, polx const *matrix, polx const *vector, size_t rows, size_t cols)
{
    polxvec_setzero(result, rows);
    polx_matmul_add(result, matrix, vector, rows, cols);
}

// todo: this can probably be replaced by polxvec_mul_extension which is probably more performant
// result += matrix^T * vector
void polx_matmul_trans_add(polx *result, polx const *matrix, polx const *vector, size_t rows, size_t cols)
{
    for (size_t j = 0; j != cols; j++) {
        for (size_t i = 0; i != rows; i++) {
            polx_mul_add(result + j, matrix + i*cols + j, vector + i);
        }
    }
}


// result = matrix^T * vector
void polx_matmul_trans(polx *result, polx const *matrix, polx const *vector, size_t rows, size_t cols)
{
    polxvec_setzero(result, cols);
    polx_matmul_trans_add(result, matrix, vector, rows, cols);
}


// convert integer mod 2^N+1 to a coefficient vector of a polynomial mod X^N+1
void lift_to_coeffs(int64_t *coeffs, mpz_t const n)
{
    mpz_t n_clone, remainder;
    mpz_inits(n_clone, remainder, NULL);
    mpz_set(n_clone, n);
    size_t length = 0;
    // non-adjacent form
    int64_t naf[N+1] = {};

    while (mpz_sgn(n_clone) != 0) {
        assert(length < N + 1);

        if (mpz_odd_p(n_clone)) {
            mpz_mod_ui(remainder, n_clone, 4);
            unsigned long rem = mpz_get_ui(remainder);

            if (rem == 1) {
                naf[length++] = 1;
            } else {
                naf[length++] = -1;
                mpz_add_ui(n_clone, n_clone, 1);
            }
        } else {
            naf[length++] = 0;
        }

        mpz_fdiv_q_2exp(n_clone, n_clone, 1);
    }

    memcpy(coeffs, naf, N * sizeof *coeffs);
    // X^{N} == -1
    coeffs[0] -= (naf[N] == 1);


    mpz_clears(n_clone, remainder, NULL);

}

void lift_to_coeffs_vector(int64_t *coeffs_vec, mpz_t const *in_vec, size_t n)
{
    for (size_t i = 0; i != n; i++) {
        lift_to_coeffs(coeffs_vec + N * i, in_vec[i]);
    }
}

void polxvec_frommpzvec(polx *vec, mpz_t const *in_vec, size_t len)
{
    int64_t *coeffs = _malloc(len*N*sizeof *coeffs);
    lift_to_coeffs_vector(coeffs, in_vec, len);
    polxvec_fromint64vec(vec, len, 1, coeffs);
    free(coeffs);

}
void polx_frommpz(polx *x, mpz_t const in)
{
    int64_t coeffs[N];
    lift_to_coeffs(coeffs, in);
    polxvec_fromint64vec(x, 1, 1, coeffs);

}


void polxvec_bitpack(uint8_t *r, polx const *a, size_t len)
{
    polz *az = _aligned_alloc(64, len * sizeof *az);
    polzvec_frompolxvec(az, a, len);
    polzvec_bitpack(r, az, len);
    free(az);

}



// debug function for importing into python
void polxvec_print_debug(char const *name, polx const *a, size_t len)
{
    mpz_t tmp, mod;
    mpz_inits(tmp, mod, NULL);
    mpz_set_str(mod, MOD_STR, 10);
    printf("%s = np.array([", name);
    for (size_t i = 0; i != len; i++) {
        polx_eval(tmp, a + i, 2, &mod);
        gmp_printf("%Zd, ", tmp);
    }
    printf("], dtype='object')\n");
    mpz_clears(tmp, mod, NULL);

}

typedef struct {
    // round1
    mpz_t *psis[ELL];

    // round2 
    mpz_t *alphas[ELL];
    mpz_t *betas[ELL];
    mpz_t *gammas[ELL];
    mpz_t *deltas[ELL][ELL];

    // round3
    polx *taus[3];
    polx *omegas[ELL2];
    int64_t chis[ELL2][ELL][N];
} challenges;

void init_challenges(challenges *challs, R1CSParams const *rp)
{
    challs->psis[0] = new_mpz_array(ELL*ELL*rp->k + 4*ELL*rp->k);
    challs->alphas[0] = challs->psis[0] + ELL*rp->k;
    challs->betas[0] = challs->alphas[0] + ELL*rp->k;
    challs->gammas[0] = challs->betas[0] + ELL*rp->k;
    challs->deltas[0][0] = challs->gammas[0] + ELL*rp->k;
    for (size_t i = 1; i != ELL; i++) {
        challs->deltas[0][i] = challs->deltas[0][i-1] + rp->k;
    }

    for (size_t i = 1; i != ELL; i++) {
        challs->psis[i] = challs->psis[i-1] + rp->k; 
        challs->alphas[i] = challs->alphas[i-1] + rp->k;
        challs->betas[i] = challs->betas[i-1] + rp->k;
        challs->gammas[i] = challs->gammas[i-1] + rp->k;
        challs->deltas[i][0] = challs->deltas[i-1][ELL-1] + rp->k;
        for (size_t j = 1; j != ELL; j++) {
            challs->deltas[i][j] = challs->deltas[i][j-1] + rp->k;
        }
    }
    challs->taus[0] = _aligned_alloc(64, (rp->m[0] + rp->m[1] + rp->m[2] + ELL2*rp->eval_vec_len) * sizeof (polx));
    challs->taus[1] = challs->taus[0] + rp->m[0];
    challs->taus[2] = challs->taus[1] + rp->m[1];
    challs->omegas[0] = challs->taus[2] + rp->m[2];
    for (size_t i = 1; i != ELL2; i++) {
        challs->omegas[i] = challs->omegas[i-1] + rp->eval_vec_len;
    }
}

void free_challenges(challenges *challs, R1CSParams const *rp)
{
    free_mpz_array(challs->psis[0], ELL*ELL*rp->k + 4*ELL*rp->k);
    free(challs->taus[0]);
}

void squeeze_mpz(mpz_t result, shake128incctx *shakectx)
{
    uint8_t buf[N/8];
    shake128_inc_squeeze(buf, N/8, shakectx);
    mpz_import(result, N/8, 1, 1, 0, 0, buf);
}

void prepare_challenge_hash(shake128incctx *shakectx, uint8_t *h, polx const *commitment, size_t comm_size, uint8_t domain)
{
    uint8_t *hashbuf = _aligned_alloc(16, comm_size * N * QBYTES);
    polxvec_bitpack(hashbuf, commitment, comm_size);
    shake128_inc_init(shakectx);
    shake128_inc_absorb(shakectx, &domain, 1);
    shake128_inc_absorb(shakectx, h, 16);
    shake128_inc_absorb(shakectx, hashbuf, comm_size * N * QBYTES);
    shake128_inc_finalize(shakectx);
    shake128_inc_squeeze(h, 16, shakectx);
    free(hashbuf);
}

void get_round1_challenges(challenges *challs, uint8_t *h, polx const *commitment, R1CSParams const *rp)
{
    shake128incctx shakectx;
    prepare_challenge_hash(&shakectx, h, commitment, rp->m[0], 1);

    for (size_t i = 0; i != ELL; i++) {
        for (size_t j = 0; j != rp->k; j++) {
            squeeze_mpz(challs->psis[i][j], &shakectx);
        }
    }
}
void get_round2_challenges(challenges *challs, uint8_t *h, polx const *commitment, R1CSParams const *rp)
{
    shake128incctx shakectx;
    prepare_challenge_hash(&shakectx, h, commitment, rp->m[1], 2);

    // Taking advantage of the fact that all the mpz_t challenges are part of the same underlying buffer
    // So this initializes alphas,betas,gammas,deltas
    for (size_t j = 0; j != ELL*(3+ELL)*rp->k; j++) {
        squeeze_mpz(challs->alphas[0][j], &shakectx);
    }
}

void get_round3_challenges(challenges *challs, uint8_t *h, polx const *commitment, R1CSParams const *rp)
{
    shake128incctx shakectx;
    uint64_t nonce = 1ul << 16;

    prepare_challenge_hash(&shakectx, h, commitment, rp->m[2], 3);

    polxvec_challenge(challs->taus[0], rp->m[0] + rp->m[1] + rp->m[2], h, nonce++);
    polxvec_almostuniform(challs->omegas[0], rp->eval_vec_len*ELL2, h, nonce++);

    //polx z_vec[ELL];
    poly z_vec[ELL];
    for (size_t i = 0; i != ELL2; i++) {
        // These challenges shouldn't be ternary, fix this!!!
        // Uniform challenges cause polx overflow representation in sparsecnst_check
        //polzvec_uniform(z_vec, ELL, h, nonce++);
        polyvec_ternary(z_vec, ELL, h, nonce++);
        for (size_t j = 0; j != ELL; j++) {
            for (size_t z = 0; z != N; z++) {
                challs->chis[i][j][z] = z_vec[j].vec->c[z]+1;
            }
        }
    }
}




void mpz_mul_vec(mpz_t *result, mpz_t const *a, mpz_t const *b, size_t len)
{
    for (size_t i = 0; i != len; i++) {
        mpz_mul(result[i], a[i], b[i]);
    }
}

void mpz_add_vec(mpz_t *result, mpz_t const *a, mpz_t const *b, size_t len)
{
    for (size_t i = 0; i != len; i++) {
        mpz_add(result[i], a[i], b[i]);
    }
}

void mpz_addmul_vec(mpz_t *result, mpz_t const *a, mpz_t const *b, size_t len)
{
    for (size_t i = 0; i != len; i++) {
        mpz_addmul(result[i], a[i], b[i]);
    }
}

void mpz_neg_vec(mpz_t *result, mpz_t const *a, size_t len)
{
    for (size_t i = 0; i != len; i++) {
        mpz_neg(result[i], a[i]);
    }
}

void mpz_sub_vec(mpz_t *result, mpz_t const *a, mpz_t const *b, size_t len)
{
    for (size_t i = 0; i != len; i++) {
        mpz_sub(result[i], a[i], b[i]);
    }
}

void mpz_mod_vec(mpz_t *result, mpz_t const *a, mpz_t const mod, size_t len)
{
    for (size_t i = 0; i != len; i++) {
        mpz_mod(result[i], a[i], mod);
    }
}


void f_r1cs(prncplstmnt *st, mpz_sparsemat const *A, mpz_sparsemat const *B, mpz_sparsemat const *C, challenges const *challs, mpz_t const mod)
{


    poly one = {};
    one.vec->c[0] = 1;
    polx onex;
    polx_frompoly(&onex, &one);


    mpz_t *scratch = new_mpz_array(MAX(st->n[0],st->n[1]));
    for (size_t i = 0; i != ELL; i++) {
        // QUADRATIC constraints
        // 1 * <b, d_i>
        st->cnst[i].a->len = 1;
        st->cnst[i].a->rows[0] = 2;
        st->cnst[i].a->cols[0] = 4+i;
        st->cnst[i].a->coeffs[0] = onex;



        // LINEAR constraints

        // \phi_0^{(i)} = A^T \alpha^{(i)} + B^T \beta^{(i)} + C^T \gamma^{(i)}
        // corresponds to wt.s[0] = w
        sparsematmul_trans(scratch, A, challs->alphas[i], st->n[0], mod);
        sparsematmul_trans_add(scratch, B, challs->betas[i], mod);
        sparsematmul_trans_add(scratch, C, challs->gammas[i], mod);
        polxvec_frommpzvec(st->cnst[i].phi[0], scratch, st->n[0]);


        // phi_1^{(i)} = -\alpha^{(i)} + \sum_{j=0}^l psi_j \circ \delta_j^{(i)}
        // corresponds to wt.s[1] = a = Aw
        mpz_mul_vec(scratch, challs->psis[0], challs->deltas[i][0], st->n[1]);
        for (size_t j = 1; j != ELL; j++) {
            mpz_addmul_vec(scratch, challs->psis[j], challs->deltas[i][j], st->n[1]);
        }
        mpz_sub_vec(scratch, scratch, challs->alphas[i], st->n[1]);
        mpz_mod_vec(scratch, scratch, mod, st->n[1]);
        polxvec_frommpzvec(st->cnst[i].phi[1], scratch, st->n[1]);


        // phi_2^{(i)} = -\beta^{i}
        // corresponds to wt.s[2] = b = Bw
        mpz_neg_vec(scratch, challs->betas[i], st->n[2]);
        mpz_mod_vec(scratch, scratch, mod, st->n[2]);
        polxvec_frommpzvec(st->cnst[i].phi[2], scratch, st->n[2]);

        // phi_3^{(i)} = -\gamma^{(i)} - \psi_i
        // corresponds to wt.s[3] = c = Cw
        mpz_neg_vec(scratch, challs->gammas[i], st->n[3]);
        mpz_sub_vec(scratch, scratch, challs->psis[i], st->n[3]);
        mpz_mod_vec(scratch, scratch, mod, st->n[3]);
        polxvec_frommpzvec(st->cnst[i].phi[3], scratch, st->n[3]);


        // for all j \in [ELL] \phi_{4+j}^((i)} = - delta_j
        // corresponds to wt.s[4+i] = d_i = \psi_i \circ a = \psi_i \circ Aw
        for (size_t j = 0; j != ELL; j++) {
            mpz_neg_vec(scratch, challs->deltas[i][j], st->n[4+j]);
            mpz_mod_vec(scratch, scratch, mod, st->n[4+j]);
            polxvec_frommpzvec(st->cnst[i].phi[4+j], scratch, st->n[4+j]);
        }
    }
    free_mpz_array(scratch, MAX(st->n[0],st->n[1]));

}

// Only need this commitment check in the first constraint?
void f_comm(prncplstmnt *st, challenges const *challs, polx const *commitments, polx const *const *T1, polx const *const *T2, polx const *const *T3, R1CSParams const *rp)
{
    for (size_t j = 0; j != 4; j++) {
        polx_matmul_trans_add(st->cnst[0].phi[j], T1[j], challs->taus[0], rp->m[0], st->n[j]);
    }
    for (size_t j = 0; j != ELL; j++) {
        polx_matmul_trans_add(st->cnst[0].phi[4+j], T2[j], challs->taus[1], rp->m[1], st->n[4+j]);
    }
    for (size_t j = 0; j != EVALVECS; j++) {
        polx_matmul_trans_add(st->cnst[0].phi[4+ELL+j], T3[j], challs->taus[2], rp->m[2], st->n[4+ELL+j]);
    }

    // Now add random linear combination of commitments to constant
    polx const *comm = commitments;
    for (size_t j = 0; j != 3; j ++) {
        polxvec_sprod_add(st->cnst[0].b, comm, challs->taus[j], rp->m[j]);
        comm += rp->m[j];
    }
}

// should be the first constant-term constraint function called because this overwrites the phi vector!!
void f_conj(prncplstmnt *st, challenges const *challs)
{
    //polx *sigma_chall_buf = _aligned_alloc(64, rp->g_bw *  ELL * sizeof *sigma_chall_buf);
    for (size_t i = 0; i != ELL2; i++) {
        polxvec_sigmam1(st->cnst[i+ELL].phi[0], challs->omegas[i], st->n[st->cnst[i+ELL].idx[0]]);
        polxvec_neg(st->cnst[i+ELL].phi[1], challs->omegas[i], st->n[st->cnst[i+ELL].idx[1]]);
    }
}

// Check that last two witness vectors are binary
void f_bin(prncplstmnt *st)
{
    polx onex = {};
    poly one = {};
    one.vec->c[0] = 1;
    polx_frompoly(&onex, &one);

    polx all_onesx = {};
    poly all_ones = {};
    for (size_t i = 0; i != N; i++) {
        all_ones.vec->c[i] = 1;
    }
    polx_frompoly(&all_onesx, &all_ones);


    for (size_t i = 0; i != ELL2; i++) {
        st->cnst[i+ELL].a->len = 1;
        st->cnst[i+ELL].a->rows[0] = R1CSVECS;
        st->cnst[i+ELL].a->cols[0] = R1CSVECS+1;
        st->cnst[i+ELL].a->coeffs[0] = onex;
        for (size_t j = 0; j != st->n[R1CSVECS]; j++) {
            polx_sub(st->cnst[i+ELL].phi[1]+j, st->cnst[i+ELL].phi[1]+j, &all_onesx);
        }
    }
}

void gdgt_coeff_vec_add(int64_t *coeffs, size_t m, size_t stride, int64_t s)
{
    uint64_t base = 1;
    for (size_t i = 0; i != m; i++) {
        *coeffs += base * s;
        base <<= 1;
        coeffs += stride;
    }
}

// At some point it would be nice to write this using AVX-512 instructions
void f_eval(prncplstmnt *st, challenges const *challs, R1CSParams const *rp)
{
    // I think < 63 should be sufficient to prevent overflow but
    // give me a couple of bits as a safety mechanism
    // more mod reductions could relax this
    //assert(significant_bits(rp->carry_eqn_bound) + LOGQ <= 60);
    size_t phi_len = st->n[st->cnst[ELL].idx[1]];

    int64_t *phi_coeffs = _malloc(phi_len * N * sizeof *phi_coeffs);

    // regions of phi_coeffs
    int64_t *g_bin = phi_coeffs;
    int64_t *quot_bin = g_bin + ELL*rp->g_bw*N;
    int64_t *carry_bin = quot_bin + ceildiv(rp->v_bw * ELL, N)*N;

    int64_t q = zz_toint64(&modulus.q);
    polx tmp_b = {};
    polx *tmp_phi = _aligned_alloc(64, phi_len * sizeof *tmp_phi);
    int64_t b[N] = {};

    // For each sparse constraint (repition for soundness...)
    for (size_t i = 0; i != ELL2; i++) {
        b[0] = 0;
        memset(phi_coeffs, 0, N*phi_len * sizeof *phi_coeffs);
        // For each g_i...
        // we are going to take a random linear combination of N "carry equations"
        // Refer to the zth carry equation for g_j as the (j,z)th carry equation.
        // This corresponds with ths scalar challs[i].round3.chi[j*N+z]
        for (size_t j = 0; j != ELL; j++) {
            //for (size_t z = 0; z != N; z++) {
            for (size_t z = 0; z != N; z++) {
                // hit the yth bit of the zth coefficient of g_j with 2**y * chi[j*N+z] to add the zth coefficient of g_j to the (j,z)th carry equation
                gdgt_coeff_vec_add(g_bin + j*rp->g_bw*N + z, rp->g_bw, N, challs->chis[i][j][z]);
                // TODO: consider doing fewer modulo reductions when possible
                // OTOH figure out if G*challs->chis[i].round3.chi[j*N+z] overflow 63 bits for large moduli??
                // Add + rp->G - AC[z] to the (j,z)th carry equation
                b[0] = (b[0] + (rp->G[z] - rp->gnorm - rp->ac[z]) * challs->chis[i][j][z]) % q;
            }
            // The RHS of the (j,z)th equation has v_j * p_z
            // where p = [1, 0,...,0,2] and v_j is g_j(2)/p(2) (note: p(2) = 2^d+1)
            // Hence we only have to add terms to the (j,0)th and (j, N-1)th
            // equations .

            gdgt_coeff_vec_add(quot_bin + j * rp->v_bw, rp->v_bw, 1, -1*challs->chis[i][j][0]);
            b[0] = (b[0] + rp->vnorm * challs->chis[i][j][0]) % q;

            gdgt_coeff_vec_add(quot_bin + j * rp->v_bw, rp->v_bw, 1, -2*challs->chis[i][j][N-1]);
            b[0] = (b[0] + 2*rp->vnorm * challs->chis[i][j][N-1]) % q;

            // Now onto the carries.
            // The (j,0)th equation does not have a carry in so we add its carry out here
            gdgt_coeff_vec_add(carry_bin + rp->h_bw*(N-1)*j, rp->h_bw, 1, -2*challs->chis[i][j][0]);
            // (j,1)...(j,N-2)th equations have carry-ins and carry-outs
            for (size_t z = 1; z != N-1; z++) {
                gdgt_coeff_vec_add(carry_bin + rp->h_bw*((N-1)*j + (z-1)), rp->h_bw, 1, challs->chis[i][j][z]);
                gdgt_coeff_vec_add(carry_bin + rp->h_bw*((N-1)*j + z), rp->h_bw, 1, -2*challs->chis[i][j][z]);
            }
            // (j,N-1)th equation has a carry in as normal ...
            gdgt_coeff_vec_add(carry_bin + rp->h_bw*((N-1)*j + (N-2)), rp->h_bw, 1, challs->chis[i][j][N-1]);
            // But a fixed carry out that's stored in ac[N]
            b[0] = (b[0] - 2*rp->ac[N] * challs->chis[i][j][N-1]);
        }
        //b[0] = (b[0] + q) % q;
        polxvec_fromint64vec(&tmp_b, 1, 1, b);
        polx_sub(st->cnst[ELL+i].b, st->cnst[ELL+i].b, &tmp_b);
        polxvec_fromint64vec(tmp_phi, phi_len, 1, phi_coeffs);
        polxvec_add(st->cnst[ELL+i].phi[1], st->cnst[ELL+i].phi[1], tmp_phi, phi_len);
    }
    free(tmp_phi);
    free(phi_coeffs);
}

void polzvec_nudge(polz *r, polz const *a, size_t len, int64_t nudge)
{
    zz nudgezz;
    polz nudge_polz;
    zz_fromint64(&nudgezz, nudge);

    for (size_t i = 0; i != N; i++) {
        polz_setcoeff(&nudge_polz, &nudgezz, i);
    }
    for (size_t i = 0; i != len; i++) {
        r[i] = a[i];
        polz_center(r+i);
        polz_add(r+i,r+i, &nudge_polz);
    }
}

void polzvec_bin_decompose(poly *r, polz const *a, size_t len, uint64_t width)
{
    assert(width < LOGQ - 1);
    for (size_t i = 0; i != len; i++) {
        for (size_t j = 0; j != N; j++) {
            int64_t coeff = polz_getcoeff_int64(a+i, j);
            assert(coeff > 0);
            for (size_t k = 0; k != width; k++) {
                r[i*width+k].vec->c[j] =  coeff & 1;
                coeff >>= 1;
            }
        }
    }
}

void mpzvec_bin_decompose(poly *r, mpz_t const *input, size_t len, uint64_t width)
{
    assert(width < LOGQ - 1);
    size_t nbits = width * len;
    // could do this w/ less dynamic allocation but who cares

    size_t bit_idx = 0;
    for (size_t i = 0; i != nbits / N; i++) {
        for (size_t j = 0; j != N; j++) {
            r[i].vec->c[j] = mpz_tstbit(input[bit_idx / width], bit_idx % width);
            bit_idx++;
        }
    }

    for (size_t j = 0; j != nbits % N; j++) {
        r[nbits/N].vec->c[j] = mpz_tstbit(input[bit_idx / width], bit_idx % width);
        bit_idx++;
    }
}


// returns vector of constant polynomials (1, 2, ..., 2^{m-1})
void polxvec_gdgt(polx *r, size_t m)
{
    int64_t coeffs[N] = {};
    coeffs[0] = 1;
    for (size_t i = 0; i != m; i++) {
        polxvec_fromint64vec(r + i, 1, 1, coeffs);
        coeffs[0] <<= 1;
    }
}

// Add -g_i to st.cnst[i] for all i in ELL
void f_gdecomp(prncplstmnt *st, R1CSParams const *rp)
{
    int64_t nudge_coeffs[N];
    for (size_t i = 0; i != N; i++) {
        nudge_coeffs[i] = -rp->gnorm;
    }
    polx nudge_polx;
    polxvec_fromint64vec(&nudge_polx, 1, 1, nudge_coeffs);

    for (size_t i = 0; i != ELL; i++) {
        polx *gdgt = st->cnst[i].phi[4+ELL] + i*rp->g_bw;
        polxvec_gdgt(gdgt, rp->g_bw);
        polxvec_neg(gdgt, gdgt, rp->g_bw);
        polx_add(st->cnst[i].b, st->cnst[i].b, &nudge_polx);
    }
}


// Initialize Chihuahua statement st, witness wt, sx, and offsets
// sx will the polx version of wt->s. Both sx and wt->s are allocated as one big buffer stored in sx[0] and wt->s[0]
// such that sx[i] = sx[0] + offsets[i] and wt.s[i] = wt.s[0] + offsets[i]
void init_r1cs_stmnt_wit(prncplstmnt *st, witness *wt, polx **sx, size_t *offsets, R1CSParams const *rp)
{
    size_t wit_lens[NWITVECS] = {};

    // first witness (w) is length n, rest are k
    wit_lens[0] = rp->n;
    for (size_t i = 1; i != R1CSVECS; i++) {
        wit_lens[i] = rp->k;
    }

    // Then we have our two binary vectors for the mod (2^d+1) evaluation
    wit_lens[R1CSVECS] = wit_lens[R1CSVECS+1] = rp->eval_vec_len;

    size_t buflen = 0;
    for (size_t i = 0; i != NWITVECS; i++) {
        offsets[i] = buflen;
        buflen += wit_lens[i];
    }

    poly *wit_vecs = _aligned_alloc(64, buflen * sizeof *wit_vecs);

    init_witness_raw_buf(wt, NWITVECS, wit_lens, wit_vecs);

    // sx[i] will be the polx version of of wt->s[i]
    sx[0] = _aligned_alloc(64, buflen * sizeof **sx);
    for (size_t i = 0; i != NWITVECS; i++) {
        sx[i] = sx[0] + offsets[i];
    }

    size_t idx[NWITVECS];
    for (size_t i = 0; i != NWITVECS; i++) {
        idx[i] = i;
    }

    uint64_t betasq = beta_squared(rp);

    init_prncplstmnt_raw(st, NWITVECS, wit_lens, betasq, ELL+ELL2, 1);
    for (size_t i = 0; i != ELL; i++) {
        init_sparsecnst_raw(st->cnst + i, NWITVECS, NWITVECS, idx, wit_lens, 1, true, false);
    }

    // For the const term constraints, only our evalvecs are non-zero
    size_t idx_const[EVALVECS] = {};
    for (size_t i = 0; i != EVALVECS; i++) {
        idx_const[i] = R1CSVECS+i;
    }

    for (size_t i = ELL; i != ELL+ELL2; i++) {
        init_sparsecnst_raw(st->cnst + i, NWITVECS, EVALVECS, idx_const, wit_lens+R1CSVECS, 0, true, false);
    }

}


void check_carry_equations(polz const *gs_z, mpz_t const *carries, mpz_t const *quotients, R1CSParams const *rp)
{
    mpz_t lhs, rhs;
    mpz_inits(lhs,rhs,NULL);
    int64_t p[N] = {};
    p[0] = 1;
    p[N-1] = 2;
    for (size_t i = 0; i != ELL; i++) {
        mpz_t carry_in;
        mpz_init_set_ui(carry_in, 0);
        for (size_t j = 0; j != N; j++) {
            int64_t gij = polz_getcoeff_int64(gs_z +i, j);
            mpz_set_si(lhs, gij);
            mpz_add_ui(lhs, lhs, rp->G[j]);
            mpz_add(lhs,lhs, carry_in);
            mpz_mul_ui(rhs, quotients[i], p[j]);
            if (j != N-1) {
                mpz_addmul_ui(rhs, carries[i*(N-1) + j], 2);
                mpz_set(carry_in, carries[i*(N-1) + j]);
            } else {
                mpz_add_ui(rhs, rhs, 2*rp->ac[N]);
            }
            mpz_add_ui(rhs,rhs, rp->ac[j]);
            assert(mpz_cmp(lhs, rhs) == 0);
        }
    }
}


// generate proof for Aw \circ Bw = Cw (mod 'mod')
// T1,T2,T3 are commitment matrices.
// A,B,C have dimension rp->k x rp->n
void r1cs_reduction(mpz_sparsemat const *A, mpz_sparsemat const *B, mpz_sparsemat const *C, polx const *const *T1, polx const *const *T2, polx const *const *T3, mpz_t *w, mpz_t const mod, uint8_t *hashstate, R1CSParams const *rp)
{
    //struct timespec start, end;
    //clock_gettime(CLOCK_MONOTONIC, &start);

    // wt.s[i] = &wt.s[0] + offsets[i]
    size_t offsets[NWITVECS] = {};
    prncplstmnt st = {};
    witness wt = {};
    // polx versions of witness vectors stored in wt.s
    polx *sx[NWITVECS] = {};

    init_r1cs_stmnt_wit(&st, &wt, sx, offsets, rp);

    // compute Aw Bw Cw, store contiguously
    mpz_t *mat_prods = new_mpz_array(offsets[4] - offsets[1]);
    sparsematmul(mat_prods, A, w, st.n[1], mod);
    sparsematmul(mat_prods + st.n[1], B, w, st.n[2], mod);
    sparsematmul(mat_prods + st.n[1] + st.n[2], C, w, st.n[3], mod);

    // coefficient vectors for w||Aw||Bw||Cw
    // TODO: include binary check in labrador constraints
    int64_t *encoded = _malloc(offsets[4] * N * sizeof *encoded);
    lift_to_coeffs_vector(encoded, w, st.n[0]);
    for (size_t i = 1; i != 4; i++) {
        lift_to_coeffs_vector(encoded + offsets[i] * N, mat_prods + (i-1)*st.n[i], st.n[i]);
    }

    for (size_t i = 0; i != 4; i++) {
        polyvec_fromint64vec(wt.s[i], st.n[i], 1, encoded + offsets[i]*N);
    }
    free(encoded);

    for (size_t i = 0; i != 4; i++) {
        polxvec_frompolyvec(sx[i], wt.s[i], st.n[i]);
    }

    // First commitment
    // TODO: include commitment check in labrador constraints
    polx *commitments = _aligned_alloc(64, (rp->m[0] + rp->m[1] + rp->m[2]) * sizeof *commitments);

    polx_matmul(commitments, T1[0], sx[0], rp->m[0], st.n[0]);

    for (size_t i = 1; i != 4; i++) {
        polx_matmul_add(commitments, T1[i], sx[i], rp->m[0], st.n[i]);
    }

    // TODO also hash public rp into hashstate at somepoint
    challenges challs[1];
    init_challenges(challs, rp);
    get_round1_challenges(challs, hashstate, commitments, rp);

    // ds[i] = psi[i] * Aw[i]
    size_t ds_len = offsets[4+ELL] - offsets[4];
    mpz_t *ds = new_mpz_array(ds_len);
    for (size_t i = 0; i != ELL; i++) {
        size_t idx = offsets[4+i] - offsets[4];
        mpz_mul_vec(ds + idx, challs->psis[i], mat_prods, st.n[4+i]);
        mpz_mod_vec(ds + idx, ds + idx, mod, st.n[4+i]);
    }
    free_mpz_array(mat_prods, offsets[4] - offsets[1]);

    // coeff vectors of ds[i]
    int64_t *d_coeffs = _malloc(ds_len*N*sizeof *d_coeffs);
    lift_to_coeffs_vector(d_coeffs, ds, ds_len);
    free_mpz_array(ds, ds_len);
    // poly representation of ds
    //poly *wit_vecs2 = wit_vecs + (3*rp->k+rp->n);
    for (size_t i = 0; i != ELL; i++) {
        polyvec_fromint64vec(wt.s[4+i], st.n[4+i], 1, d_coeffs + (offsets[4+i] - offsets[4])*N);
        polxvec_frompolyvec(sx[4+i], wt.s[4+i], st.n[4+i]);
    }
    free(d_coeffs);

    // second commitment
    polx *commitment2 = commitments + rp->m[0];
    polxvec_setzero(commitment2, rp->m[1]);
    for (size_t i = 0; i != ELL; i++) {
        polx_matmul_add(commitment2, T2[i], sx[4+i], rp->m[1], st.n[4+i]);
    }

    get_round2_challenges(challs, hashstate, commitment2, rp);

    // In principle, we'd do round 3 right now, but we need to compute the g_is
    // So we take a quick break to construct princplstmnt since we have to do that
    // later anyway, and we'll partiallly initalize it with f_r1cs and
    // then call sparsecnst_eval to compute the gs. Then we'll proceed with round3

    // num witness vecs


    f_r1cs(&st, A, B, C, challs, mod);

    polx *gs = _aligned_alloc(64, ELL * sizeof *gs);
    polz *gs_z = _aligned_alloc(64, ELL * sizeof *gs);
    mpz_t *quotients = new_mpz_array(ELL);
    mpz_t remainder;
    mpz_init(remainder);
    for (size_t i = 0; i != ELL; i++) {
        // Now we evaluate f_r1cs. Set st.cnst[i].nz to R1CSVECS for performance
        size_t old_nz = st.cnst[i].nz;
        st.cnst[i].nz = R1CSVECS;
        sparsecnst_eval(gs + i, &st.cnst[i], sx, &wt);
        st.cnst[i].nz = old_nz;
        polx_eval(quotients[i], gs + i, 2, NULL);
        mpz_fdiv_qr(quotients[i], remainder, quotients[i], mod);
        assert(mpz_sgn(remainder) == 0);
    }
    mpz_clear(remainder);

    polzvec_frompolxvec(gs_z, gs, ELL);
    polzvec_center(gs_z, ELL);

    // compute carries
    mpz_t *carries = new_mpz_array(ELL*(N-1));
    for (size_t i = 0; i != ELL; i++) {
        for (size_t j = 0; j != (N-1); j++) {
            size_t carry_idx = i*(N-1)+j;
            mpz_set_si(carries[carry_idx], polz_getcoeff_int64(gs_z + i, j));
            mpz_add_ui(carries[carry_idx], carries[carry_idx], rp->G[j]);
            if (j == 0) {
                mpz_sub(carries[carry_idx], carries[carry_idx], quotients[i]);
            } else {
                mpz_add(carries[carry_idx], carries[carry_idx], carries[carry_idx-1]);
            }
            assert(mpz_sgn(carries[carry_idx]) >= 0);
            mpz_fdiv_q_ui(carries[carry_idx], carries[carry_idx], 2);
            assert(mpz_sizeinbase(carries[carry_idx],2) <= rp->h_bw);
        }
    }

    // We need to make gs and quotients non-negative before we binary decompose them
    // This is "subtracted off" in f_eval
    polzvec_nudge(gs_z, gs_z, ELL, rp->gnorm);
    polzvec_center(gs_z, ELL);
    for (size_t i = 0; i != ELL; i++) {
        mpz_add_ui(quotients[i], quotients[i], rp->vnorm);
    }


    // BEGIN ROUND3
    polzvec_bin_decompose(wt.s[R1CSVECS], gs_z, ELL, rp->g_bw);
    mpzvec_bin_decompose(wt.s[R1CSVECS] + rp->g_bw * ELL, quotients, ELL, rp->v_bw);
    mpzvec_bin_decompose(wt.s[R1CSVECS] + rp->g_bw * ELL + ceildiv(ELL*rp->v_bw, N), carries, ELL*(N-1), rp->h_bw);

    polyvec_sigmam1(wt.s[R1CSVECS+1], wt.s[R1CSVECS], st.n[R1CSVECS+1]);

    for (size_t i = R1CSVECS; i != NWITVECS; i++) {
        polxvec_frompolyvec(sx[i], wt.s[i], st.n[i]);
    }


    polx *commitment3 = commitment2 + rp->m[1];
    polx_matmul(commitment3, T3[0], sx[R1CSVECS], rp->m[2], st.n[R1CSVECS]);
    polx_matmul_add(commitment3, T3[1], sx[R1CSVECS+1], rp->m[2], st.n[R1CSVECS+1]);

    // Now that we have c3 = T3(g||g'||...||), we can generate the rest of the challs
    get_round3_challenges(challs, hashstate, commitment3, rp);

    // Now we add the rest of the constraints

    // add <g, -gdgt_i> to each f_i
    f_gdecomp(&st, rp);
    // add commitment consistency constraints
    f_comm(&st, challs, commitments, T1, T2, T3, rp);

    // constant term constraints
    // f_conj must come first because it overwrites st.cnst[ELL+i].phis
    f_conj(&st, challs);
    f_eval(&st, challs, rp);
    f_bin(&st);



    memcpy(st.h, hashstate, 16);
    for (size_t i = 0; i != ELL; i++) {
        sparsecnst_hash(st.h, st.cnst + i, st.n);
    }
    for (size_t i = ELL; i != ELL+ELL2; i++) {
        sparsecnst_hash(st.h, st.cnst + i, st.n);
    }

    print_prncplstmnt_pp(&st);
    // should pass trivially per above
    assert(principle_verify(&st, &wt) == 0);

    uint64_t running_norm = 0;
    for (size_t i = 0; i != wt.r; i++) {
        wt.normsq[i] = polyvec_sprodz(wt.s[i], wt.s[i], wt.n[i]);
        running_norm += wt.normsq[i];
    }
    assert(running_norm <= beta_squared(rp));

    composite cproof = {};
    assert(composite_prove_principle(&cproof, &st, &wt) == 0);

    //clock_gettime(CLOCK_MONOTONIC, &end);
    //long seconds = end.tv_sec - start.tv_sec;
    //long nano = end.tv_nsec - start.tv_nsec;
    //double ms = seconds * 1e6 + nano / 1e3;
    //printf("duration:%lf\n", ms);
    printf("%d\n", composite_verify_principle(&cproof, &st));

    free(commitments);
    free_challenges(challs, rp);
    free_witness(&wt); // frees wit_vecs
    free_prncplstmnt(&st);
    free(gs);
    free(gs_z);
    free_mpz_array(quotients, ELL);
    free_mpz_array(carries, ELL*(N-1));
    free(sx[0]);
}


int main(void)
{
    R1CSParams rp = {};
    new_r1cs_params(&rp, 1000,100, (size_t [3]) {
        20,20,20 // commitment sizes, not secure, just placeholders
    });

    gmp_randstate_t grand;
    gmp_randinit_default(grand);
    mpz_t *w = new_mpz_array(rp.n);
    for (size_t i = 0; i != rp.n; i++) {
        mpz_urandomb(w[i], grand, N);
    }

    mpz_sparsemat A, B, C;
    init_mpz_sparsemat(&A, rp.n);
    init_mpz_sparsemat(&B, rp.n);
    init_mpz_sparsemat(&C, rp.n);

    for (size_t i = 0; i != rp.n; i++) {
        add_entry_ui(&A, i, i, 1);
        add_entry_ui(&B, i, i, 1);
        add_entry(&C, i, i, w[i]);
    }


    mpz_t mod;
    mpz_init(mod);
    mpz_set_str(mod, MOD_STR, 10);

    size_t t1_size = (3*rp.k+rp.n)*rp.m[0];
    size_t t2_size = rp.m[1] * ELL*rp.k;
    size_t t3_size = rp.m[2] * 2*rp.eval_vec_len;

    polx *comm = _aligned_alloc(64, (t1_size + t2_size + t3_size) * sizeof *comm);

    __attribute__((aligned(16)))
    unsigned char seed[16] = {150, 98, 20, 81, 126, 151, 66, 43, 68, 235, 210, 118, 199, 77, 163, 30};
    int64_t nonce = 0;
    polxvec_almostuniform(comm, t1_size + t2_size + t3_size, seed, nonce);
    polx const *T1[4];
    T1[0] = comm;
    T1[1] = comm + rp.n*rp.m[0];
    for (size_t i = 2; i != 4; i++) {
        T1[i] = T1[i-1] + rp.k*rp.m[0];
    }
    polx const *T2[ELL];
    T2[0] = comm + (3*rp.k+rp.n)*rp.m[0];
    for (size_t i = 1; i != ELL; i++) {
        T2[i] = T2[i-1] + rp.k*rp.m[1];
    }
    polx const *T3[EVALVECS];
    T3[0] = T2[0] + rp.m[1] * ELL * rp.k;
    T3[1] = T3[0] + rp.eval_vec_len * rp.m[2];

    r1cs_reduction(&A, &B, &C, T1, T2, T3, w, mod, seed, &rp);
    free_mpz_sparsemat(&A);
    free_mpz_sparsemat(&B);
    free_mpz_sparsemat(&C);
    free(comm);
    mpz_clear(mod);
    free_mpz_array(w, rp.n);

}

