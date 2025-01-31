#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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


// could make this generic over data.h instantiation but not for now
#define Q 4001957325

#define CEILDIV(x,y) ((x + y - 1) / y)
#define INT_BYTES CEILDIV(N+1,8)

// update h with hash of constraints in sparsecnst
// TODO update w/ sparse representation
void sparsecnst_hash(uint8_t h[16], sparsecnst const *cnst, size_t nz,
                     const size_t n[nz], size_t deg)
{
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
    for (size_t j = 0; j != nz; j++) {
        for (size_t k = 0; k != n[j]; k++) {
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
void polz_eval(mpz_t output, const polz *t, int64_t coord, mpz_t const mod)
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
            mpz_mod(coeff, coeff, mod);
        }

        mpz_addmul(output, coeff, power);
        mpz_mod(output, output, mod);
        mpz_mul_ui(power, power, coord);
        mpz_mod(power, power, mod);
    }
    mpz_clears(power, coeff, temp, NULL);
}

void poly_eval(mpz_t output, const poly *t, int64_t coord, mpz_t const mod)
{
    polz z;
    polz_frompoly(&z, t);
    polz_center(&z);
    polz_eval(output, &z, coord, mod);
}


void polx_eval(mpz_t output, polx const *t, int64_t coord, mpz_t const mod)
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

// matrix represented as 1d array where (i,j)th cell is (matrix + i * cols + j)
void polx_matmul(polx *result, polx const *matrix, polx const *vector, size_t rows, size_t cols)
{
    polxvec_setzero(result, rows);
    for (size_t i = 0; i != rows; i++) {
        for (size_t j = 0; j != cols; j++) {
            polx_mul_add(result + i, matrix + i*cols + j, vector + j);
        }
    }
}

// result = matrix^T * vector (mod MOD)
void matmul_transpose(mpz_t *result, mpz_t *const *matrix, mpz_t const *vector, size_t rows, size_t cols, mpz_t const mod)
{
    for (size_t i = 0; i != cols; i++) {
        mpz_set_ui(result[i], 0);
    }

    mpz_t tmp;
    mpz_init(tmp);
    for (size_t j = 0; j != cols; j++) {
        for (size_t i = 0; i != rows; i++) {
	    mpz_addmul(result[j], matrix[i][j], vector[i]);
            mpz_mod(result[j], result[j], mod);
        }
    }
    mpz_clear(tmp);
}

// result = matrix * vector (mod MOD)
void matmul(mpz_t *result, mpz_t * const *matrix, mpz_t const * vector, size_t rows, size_t cols, mpz_t const mod)
{
    for (size_t i = 0; i != rows; i++) {
        mpz_set_ui(result[i], 0);
    }

    for (size_t i = 0; i != rows; i++) {
        for (size_t j = 0; j != cols; j++) {
	    mpz_addmul(result[i], matrix[i][j], vector[j]);
            mpz_mod(result[i], result[i], mod);
        }
    }
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

// TODO: get rid of this values < 2**d+1 have enough entropy for challenges
void rejection_sample(mpz_t *out, size_t n, mpz_t const mod, mpz_t cutoff, size_t len, shake128incctx *ctx)
{
    size_t i = 0;

    uint8_t *hashout = _malloc(len);
    mpz_t temp;
    mpz_init(temp);

    while (i != n) {
        shake128_inc_squeeze(hashout, len, ctx);

        if (mpz_cmp(temp, cutoff) < 0) {
            mpz_init(out[i]);
            mpz_mod(out[i], temp, mod);
            i++;
        }
    }

    mpz_clear(temp);
    free(hashout);
}


//void polx_print_eval(char const *fmt, polx *a, int64_t x, ...)
//{
//    va_list args;
//    va_start(args, x);
//    mpz_t tmp, mod;
//    mpz_inits(tmp, mod, NULL);
//    mpz_set_str(mod, MOD_STR, 10);
//    polx_eval(tmp, a, x, mod);
//    gmp_printf(fmt, tmp, args);
//    mpz_clears(tmp, mod, NULL);
//    va_end(args);
//}


// debug function for importing into python
void polxvec_print_debug(char const *name, polx const *a, size_t len)
{
    mpz_t tmp, mod;
    mpz_inits(tmp, mod, NULL);
    mpz_set_str(mod, MOD_STR, 10);
    printf("%s = np.array([", name);
    for (size_t i = 0; i != len; i++) {
        polx_eval(tmp, a + i, 2, mod);
        gmp_printf("%Zd, ", tmp);
    }
    printf("], dtype='object')\n");

}

// window into challenge
typedef struct {
    size_t k;

    mpz_t const *alpha;
    mpz_t const *beta;
    mpz_t const *gamma;
    mpz_t const *deltas[ELL];
} chall_view;

void init_chall_view(chall_view *view, mpz_t const *buf, size_t k)
{
    view->k = k;
    view->alpha = buf;
    view->beta = view->alpha + view->k;
    view->gamma = view->beta + view->k;
    view->deltas[0] = view->gamma + view->k;
    for (size_t i = 1; i != ELL; i++) {
	view->deltas[i] = view->deltas[i-1] + view->k;
    }
}


void next_challenge(chall_view *view)
{
    init_chall_view(view, view->deltas[ELL-1] + view->k, view->k);
}


mpz_t *new_mpz_array(size_t len) {
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




// generate proof for Aw \circ Bw = Cw (mod 'mod')
// C1 and C2 are commitment matrices.
// A,B,C have dimension k x n
// m, md are #rows of C1 and C2 respectively
void r1cs_reduction(mpz_t **A, mpz_t **B, mpz_t **C, polx *C1, polx *C2, mpz_t *w, mpz_t const mod, size_t k, size_t n, size_t m, size_t md)
{

    // compute Aw Bw Cw, store contiguously
    mpz_t *mat_prods = new_mpz_array(3*k);
    matmul(mat_prods, A, w, k, n, mod);
    matmul(mat_prods + k, B, w, k, n, mod);
    matmul(mat_prods + 2*k, C, w, k, n, mod);
    print_mpz_array(mat_prods, 3*k);

    // coefficient vectors for w||Aw||Bw||Cw
    // TODO: include binary check in labrador constraints
    int64_t *encoded = _malloc((n+3*k) * N * sizeof *encoded);
    lift_to_coeffs_vector(encoded, w, n);
    for (size_t i = 0; i != 3; i++) {
        lift_to_coeffs_vector(encoded + (n + i*k) * N, mat_prods + i*k, k);
    }

    // poly representations of w||Aw||Bw||Cw
    // buffer has enough space for the rest of the witnesses computed later
    poly *wit_vecs = _aligned_alloc(64, (n+3*k + ELL*k) * sizeof *wit_vecs);
    polyvec_fromint64vec(wit_vecs, n+3*k, 1, encoded);
    // polx version of wit_vecs
    polx *wit_vecsx = _aligned_alloc(64, (3*k+n + ELL*k) * sizeof *wit_vecsx);
    polxvec_frompolyvec(wit_vecsx, wit_vecs, 3*k+n);

    // First commitment
    // TODO: include commitment check in labrador constraints
    polx *t_hat = _aligned_alloc(64, (m + md) * sizeof *t_hat);
    polx_matmul(t_hat, C1, wit_vecsx, m, 3*k+n);


    shake128incctx shakectx;
    shake128_inc_init(&shakectx);

    uint8_t *hashbuf = _aligned_alloc(16, m * N * QBYTES* sizeof *hashbuf);
    polxvec_bitpack(hashbuf, t_hat, m);

    shake128_inc_absorb(&shakectx, hashbuf, N*QBYTES*m);
    // TODO also hash public params

    // keep old hash ctx around for future absorbing
    shake128incctx tmp_ctx = shakectx;
    shake128_inc_finalize(&tmp_ctx);

    uint8_t *hashout = _malloc(ELL*k*N/8*sizeof *hashout);
    shake128_inc_squeeze(hashout, ELL*k*N/8, &tmp_ctx);

    // verifier challenge
    mpz_t *psis = _malloc(ELL*k*sizeof *psis);
    // ds[i] = psi[i] * Aw[i]
    mpz_t *ds = _malloc(ELL*k*sizeof *ds);
    for (size_t i = 0; i != ELL*k; i++) {
        mpz_inits(psis[i], ds[i], NULL);
        mpz_import(psis[i], N/8, 1, 1, 0, 0, hashout + i*N/8);
        mpz_mul(ds[i], psis[i], mat_prods[i % k]);
        mpz_mod(ds[i], ds[i], mod);
    }

    // coeff vectors of ds[i]
    int64_t *d_coeffs = _malloc(ELL*k*N*sizeof *d_coeffs);
    lift_to_coeffs_vector(d_coeffs, ds, k*ELL);
    // poly representation of ds
    poly *wit_vecs2 = wit_vecs + (3*k+n);
    polyvec_fromint64vec(wit_vecs2, ELL*k, 1, d_coeffs);
    // polx representatoin of ds
    polx *wit_vecsx2 = wit_vecsx + 3*k+n;
    polxvec_frompolyvec(wit_vecsx2, wit_vecs2, ELL*k);

    // second commitment
    polx *td_hat = t_hat + m;
    polx_matmul(td_hat, C2, wit_vecsx2, md, ELL*k);

    uint8_t *hashbuf2 = _aligned_alloc(16, (N*QBYTES*md)*sizeof *hashbuf2);
    polxvec_bitpack(hashbuf2, td_hat, md);
    shake128_inc_absorb(&shakectx, hashbuf2, N*QBYTES*md);

    // keep old hash context around for futre absorbing
    tmp_ctx = shakectx;
    shake128_inc_finalize(&tmp_ctx);


    uint8_t *hashout2 = _malloc(k*(ELL+3)*ELL*N/8 * sizeof *hashout2);
    shake128_inc_squeeze(hashout2, k*(ELL+3)*ELL*N/8, &tmp_ctx);

    // final hallenges
    mpz_t *challs = _malloc(k*(ELL+3)*ELL*sizeof *challs);

    for (size_t i = 0; i != k*(ELL+3)*ELL; i++) {
        mpz_init(challs[i]);
        mpz_import(challs[i], N/8, 1, 1, 0, 0, hashout2 + i*N/8);
    }


    // BEGIN constructing stuff for chihuahua call

    // num witness vecs
    size_t r = 4 + ELL;

    size_t wit_lens[r];

    // first witness (w) is length n, rest are k
    wit_lens[0] = n;
    for (size_t i = 1; i != r; i++) {
        wit_lens[i] = k;

    }

    witness wt = {};
    init_witness_raw(&wt, r, wit_lens);
    wt.s[0] = wit_vecs;

    // polx version of wt.s already have this as wit_vecsx, but need it in 2d array form
    polx *sx[r];
    sx[0] = wit_vecsx;
    wt.normsq[0] = n * N/2;

    for (size_t i = 1; i != r; i++) {
        wt.s[i] = wt.s[i-1] + wit_lens[i-1];
        wt.normsq[i] = k * N/2;
        sx[i] = sx[i-1] + wit_lens[i-1];
    }

    uint64_t betasq = 128. / 30. * ((3+ELL)*k+n)*N/2;

    prncplstmnt st = {};
    init_prncplstmnt_raw(&st, r, wit_lens, betasq, ELL, 1);
    shake128_inc_finalize(&shakectx);
    shake128_inc_squeeze(st.h, 16, &shakectx);

    // need constant 1 polynomial
    poly one = {};
    one.vec->c[0] = 1;
    polx onex;
    polx_frompoly(&onex, &one);

    // for each sparse constraint there are 5 non-zero linear constraints
    // 1 for w, a = Aw, b = Bw, c = Cw and exactly one of the d_i = psi * a[i]s
    size_t nz = 5;

    size_t idx[nz];
    // only the last idx changes across sparse cnsts, first nz-1 all the same
    for (size_t i = 0; i != nz-1; i++) {
        idx[i] = i;
    }

    mpz_t eval;
    mpz_init(eval);

    chall_view chall = {};
    init_chall_view(&chall, challs, k);

    // scratch space for mpz operations
    mpz_t *scratch = new_mpz_array(2*n);
    for (size_t i = 0; i != ELL; i++, next_challenge(&chall)) {
        idx[nz-1] = nz-1+i;
        init_sparsecnst_raw(st.cnst + i, r, nz, idx, wit_lens, 1, true, false);

        // QUADRATIC constraints
        // 1 * <b, d_i>
        st.cnst[i].a->len = 1;
        st.cnst[i].a->rows[0] = 2;
        st.cnst[i].a->cols[0] = 4+i;
        st.cnst[i].a->coeffs[0] = onex;


        // LINEAR constraints

        // \phi_0^{(i)} = A^T \alpha^{(i)} + B^T \beta^{(i)} + C^T \gamma^{(i)}
        // corresponds to wt.s[0] = w
        matmul_transpose(scratch, A, chall.alpha, k, n, mod);
        matmul_transpose(scratch + n, B, chall.beta, k, n, mod);
	for (size_t j = 0; j != n; j++) {
	    mpz_add(scratch[j], scratch[j], scratch[j+n]);
	}
        matmul_transpose(scratch + n, C, chall.gamma, k, n, mod);
	for (size_t j = 0; j != n; j++) {
	    mpz_add(scratch[j], scratch[j], scratch[j+n]);
	    mpz_mod(scratch[j], scratch[j], mod);
	}

        polxvec_frommpzvec(st.cnst[i].phi[0], scratch, n);

        // phi_1^{(i)} = -\alpha^{(i)} + \sum_{j=0}^l psi_i \circ \delta_j^{(i)}
        // corresponds to wt.s[1] = a = Aw
	for (size_t j = 0; j != k; j++) {
	    mpz_neg(scratch[0], chall.alpha[j]);
	    for (size_t z = 0; z != ELL; z++) {
		mpz_addmul(scratch[0], psis[i*k+j], chall.deltas[z][j]);
		mpz_mod(scratch[0], scratch[0], mod);
	    }
	    polx_frommpz(st.cnst[i].phi[1] + j, scratch[0]);
	}


        // phi_2^{(i)} = -\beta^{i}
        // corresponds to wt.s[2] = b = Bw
	for (size_t j = 0; j != k; j++) {
	    mpz_neg(scratch[0], chall.beta[j]);
	    mpz_mod(scratch[0], scratch[0], mod);
	    polx_frommpz(st.cnst[i].phi[2] + j, scratch[0]);
	}

        // phi_3^{(i)} = -\gamma^{(i)} - \psi_i
        // corresponds to wt.s[3] = c = Cw
	for (size_t j = 0; j != k; j++) {
	    mpz_neg(scratch[0], chall.gamma[j]);
	    mpz_sub(scratch[0], scratch[0], psis[i*k+j]);
	    mpz_mod(scratch[0], scratch[0], mod);
	    polx_frommpz(st.cnst[i].phi[3] + j, scratch[0]);
	}

        // \phi_{4} = -\sum_{i=0}^l \delta^{(i)_j}
        // corresponds to wt.s[4+i] = d_i
	// TODO compute sum of deltas earlier and re-use
	for (size_t j = 0; j != k; j++) {
	    mpz_neg(scratch[0], chall.deltas[0][j]);
	    for (size_t z = 1; z != ELL; z++) {
		mpz_sub(scratch[0], scratch[0], chall.deltas[z][j]);
	    }
	    mpz_mod(scratch[0], scratch[0], mod);
	    polx_frommpz(st.cnst[i].phi[4] + j, scratch[0]);
	}
        //polxvec_neg(st.cnst[i].phi[4], chall.deltasx[0], k);

        //for (size_t j = 1; j != ELL; j++) {
        //    polxvec_sub(st.cnst[i].phi[4], st.cnst[i].phi[4], chall.deltasx[j], k);
        //}

        // set st.cnst[i].b to whatever it actually evaluates to
        sparsecnst_eval(st.cnst[i].b, &st.cnst[i], sx, &wt);
        sparsecnst_hash(st.h, st.cnst + i, nz, wit_lens, 1);

        // this should pass trivially because of how we set st.cnst[i].b
        assert(sparsecnst_check(st.cnst +i, sx, &wt));

        // check that st.cnst[i].b(2) == 0 mod (2**N + 1)
        polx_eval(eval, st.cnst[i].b, 2, mod);
        assert(mpz_sgn(eval) == 0);
    }
    free_mpz_array(scratch, 2*n);
    mpz_clear(eval);

    print_prncplstmnt_pp(&st);
    // should pass trivially per above
    assert(principle_verify(&st, &wt) == 0);

    composite cproof = {};
    assert(composite_prove_principle(&cproof, &st, &wt) == 0);
    assert(composite_verify_principle(&cproof, &st) == 0);
}


int main(void)
{
    // placeholders
    size_t k = 100;
    size_t n = 100;
    size_t m = 3;
    size_t md = 3;

    mpz_t **A = calloc(k, sizeof *A);

    for (size_t i = 0; i != k; i++) {
        A[i] = calloc(n, sizeof **A);

        for (size_t j = 0; j != n; j++) {
            mpz_init(A[i][j]);
            mpz_set_ui(A[i][j], i == j);
        }
    }

    mpz_t **B = calloc(k, sizeof *B);

    for (size_t i = 0; i != k; i++) {
        B[i] = calloc(n, sizeof **B);

        for (size_t j = 0; j != n; j++) {
            mpz_init(B[i][j]);
            mpz_set_ui(B[i][j], i == j);
        }
    }

    mpz_t **C = calloc(k, sizeof *C);

    for (size_t i = 0; i != k; i++) {
        C[i] = calloc(n, sizeof **C);

        for (size_t j = 0; j != n; j++) {
            mpz_init(C[i][j]);
            mpz_set_ui(C[i][j], i == j ? i : 0);
        }
    }

    mpz_t *w = calloc(n, sizeof *w);

    for (size_t i = 0; i != n; i++) {
        mpz_init(w[i]);
        mpz_set_ui(w[i], i);
    }

    mpz_t mod;
    mpz_init(mod);
    mpz_set_str(mod, MOD_STR, 10);


    polx *comm = _aligned_alloc(64, ((3*k+n)*m + md * ELL*k) * sizeof *comm);

    __attribute__((aligned(16)))
    unsigned char seed[16] = {150, 98, 20, 81, 126, 151, 66, 43, 68, 235, 210, 118, 199, 77, 163, 30};
    int64_t nonce = 0;
    polxvec_almostuniform(comm, (3*k+n)*m + md * ELL * k, seed, nonce);
    polx *C1 = comm;
    polx *C2 = comm + (3*k+n)*m;
    r1cs_reduction(A, B, C, C1, C2, w, mod, k, n, m, md);

}

