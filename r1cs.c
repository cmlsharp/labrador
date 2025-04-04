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
#define ELL 1


#define CEILDIV(x,y) ((x + y - 1) / y)
#define INT_BYTES CEILDIV(N+1,8)
#define GNORM(k,n) (((4 + ELL)*k+n)*(N/2+3))

//struct R1CSParams {
//    size_t k;
//    size_t n;
//    size_t m;
//    size_t md;
//}
//
//uint64_t gnorm(R1CSParams const *rparams)
//{
//    return ((4+ELL)*rparams->k+rparams->n)*(N/2+3)
//}
//
//uint64_t gbits_width(R1CSParams const *rparams)
//{
//    return significant_bits(gnorm(rparams))+1;
//}

void polz_print2(const polz *a) {
    for (size_t i = 0; i != N; i++) {
	int64_t x = a->limbs[0].c[i];
	for (size_t j = 1; j != L; j++) {
	    x += a->limbs[j].c[i] * (((int64_t) 1) << (14*j));
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

uint64_t significant_bits(uint64_t x)
{
    if (x == 0) return 0;
    return 64 - __builtin_clzll(x);
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

// TODO: get rid of this values < 2**d+1 have enough entropy for challenges
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
typedef struct {
    size_t k;
    size_t m;
    size_t md;
    struct {
	mpz_t *psi;
    } round1;
    struct {
	mpz_t *alpha;
	mpz_t *beta;
	mpz_t *gamma;
	mpz_t *deltas[ELL];
	polx *epsilon;
	polx *zeta;
    } round2;
} challenge;


void new_challenge(challenge *chall, size_t k, size_t m, size_t md)
{
    chall->k = k;
    chall->m = m;
    chall->md = md;
    chall->round1.psi = new_mpz_array((4 + ELL) * k);

    chall->round2.alpha = chall->round1.psi + k;
    chall->round2.beta = chall->round2.alpha + k;
    chall->round2.gamma = chall->round2.beta + k;
    chall->round2.deltas[0] = chall->round2.gamma + k;
    for (size_t i = 1; i != ELL; i ++) {
	chall->round2.deltas[i] = chall->round2.deltas[i-1] + k;
    }

    chall->round2.epsilon = _aligned_alloc(64, (m+md) * sizeof (polx));
    chall->round2.zeta = chall->round2.epsilon + m;
}

void free_challenge(challenge *chall)
{
    free_mpz_array(chall->round1.psi, (4+ELL) * chall->k);
    free(chall->round2.epsilon);
}

challenge *new_challenge_array(size_t count, size_t k, size_t m, size_t md)
{
    challenge *ret = _malloc(count * sizeof *ret);
    for (size_t i = 0; i != count; i++) {
	new_challenge(ret + i, k, m, md);
    }
    return ret;
}
void free_challenge_array(challenge *challs, size_t count)
{
    for (size_t i = 0; i != count; i++) {
	free_challenge(challs + i);
    }
}

void squeeze_mpz(mpz_t result, shake128incctx *shakectx)
{
    uint8_t buf[N/8];
    shake128_inc_squeeze(buf, N/8, shakectx);
    mpz_import(result, N/8, 1, 1, 0, 0, buf);
}

void prepare_challenge_hash(shake128incctx *shakectx, uint8_t *h, polx const *commitment, size_t comm_size)
{
    uint8_t *hashbuf = _aligned_alloc(16, comm_size * N * QBYTES);
    polxvec_bitpack(hashbuf, commitment, comm_size);
    shake128_inc_init(shakectx);
    shake128_inc_absorb(shakectx, h, 16);
    shake128_inc_absorb(shakectx, hashbuf, comm_size * N * QBYTES);
    shake128_inc_finalize(shakectx);
    shake128_inc_squeeze(h, 16, shakectx);
    free(hashbuf);
}

void get_round1_challenges(challenge *challs, uint8_t *h, polx const *commitment, size_t comm_size,  size_t n_challs)
{
    shake128incctx shakectx;
    prepare_challenge_hash(&shakectx, h, commitment, comm_size);

    size_t k = challs[0].k;
    for (size_t i = 0; i != n_challs; i++) {
	for (size_t j = 0; j != k; j++) {
	    squeeze_mpz(challs[i].round1.psi[j], &shakectx);
	}
    }
}

void get_round2_challenges(challenge *challs, uint8_t *h, polx const *commitment, size_t comm_size, size_t n_challs)
{
    shake128incctx shakectx;
    prepare_challenge_hash(&shakectx, h, commitment, comm_size);

    size_t k = challs[0].k;
    size_t m = challs[0].m;
    size_t md = challs[0].md;

    // domain separation never hurt anyone
    uint64_t nonce_domain = ((uint64_t) 1) << 16;
    for (size_t i = 0; i != n_challs; i++) {
	for (size_t j = 0; j != (3+ELL) * k; j++) {
	    squeeze_mpz(challs[i].round2.alpha[j], &shakectx);
	}
	polxvec_ternary(challs[i].round2.epsilon, m+md, h, nonce_domain + i);
    }
}



// window into challenge
//typedef struct {
//    size_t k;
//    size_t m;
//    size_t md;
//
//    mpz_t const *alpha;
//    mpz_t const *beta;
//    mpz_t const *gamma;
//    mpz_t const *deltas[ELL];
//    mpz_t const *epsilon;
//    mpz_t const *zeta;
//} chall_view;
//
//void init_chall_view(chall_view *view, mpz_t const *buf, size_t k, size_t m, size_t md)
//{
//    view->m = m;
//    view->k = k;
//    view->md = md;
//    view->alpha = buf;
//    view->beta = view->alpha + view->k;
//    view->gamma = view->beta + view->k;
//    view->deltas[0] = view->gamma + view->k;
//    for (size_t i = 1; i != ELL; i++) {
//        view->deltas[i] = view->deltas[i-1] + view->k;
//    }
//    view->epsilon = view->deltas[ELL-1] + view->k;
//    view->zeta = view->epsilon + view->m;
//}
//
//
//void next_challenge(chall_view *view)
//{
//    init_chall_view(view, view->zeta + view->md, view->k, view->m, view->md);
//}



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

uint64_t beta_squared(size_t k, size_t n)
{
    return 128. / 30. * (((3+ELL)*k+n)*(N/2+3) + N);
}

void f_r1cs(prncplstmnt *st, mpz_sparsemat const *A, mpz_sparsemat const *B, mpz_sparsemat const *C, challenge const *challenges, mpz_t const mod) {


    poly one = {};
    one.vec->c[0] = 1;
    polx onex;
    polx_frompoly(&onex, &one);


    mpz_t *scratch = new_mpz_array(MAX(st->n[0],st->n[1]));
    for (size_t i = 0; i != ELL; i++) {
	challenge chall = challenges[i];

        // QUADRATIC constraints
        // 1 * <b, d_i>
        st->cnst[i].a->len = 1;
        st->cnst[i].a->rows[0] = 2;
        st->cnst[i].a->cols[0] = 4+i;
        st->cnst[i].a->coeffs[0] = onex;



        // LINEAR constraints
        //polxvec_frommpzvec(epsilonx, chall.epsilon, m);
        //polxvec_frommpzvec(zetax, chall.zeta, md);

        // \phi_0^{(i)} = A^T \alpha^{(i)} + B^T \beta^{(i)} + C^T \gamma^{(i)} + C1[0]^T \epsilon
        // corresponds to wt.s[0] = w
        sparsematmul_trans(scratch, A, chall.round2.alpha, st->n[0], mod);
        sparsematmul_trans_add(scratch, B, chall.round2.beta, mod);
        sparsematmul_trans_add(scratch, C, chall.round2.gamma, mod);
        polxvec_frommpzvec(st->cnst[i].phi[0], scratch, st->n[0]);


        // phi_1^{(i)} = -\alpha^{(i)} + \sum_{j=0}^l psi_j \circ \delta_j^{(i)} + C1[1]^t \epsilon
        // corresponds to wt.s[1] = a = Aw
	mpz_mul_vec(scratch, challenges[0].round1.psi, chall.round2.deltas[0], st->n[1]);
	for (size_t j = 1; j != ELL; j++) {
	    mpz_addmul_vec(scratch, challenges[j].round1.psi, chall.round2.deltas[j], st->n[1]);
	}
        mpz_sub_vec(scratch, scratch, chall.round2.alpha, st->n[1]);
	mpz_mod_vec(scratch, scratch, mod, st->n[1]);
	polxvec_frommpzvec(st->cnst[i].phi[1], scratch, st->n[1]);


        // phi_2^{(i)} = -\beta^{i} + C1[2]^t \epsilon
        // corresponds to wt.s[2] = b = Bw
	mpz_neg_vec(scratch, chall.round2.beta, st->n[2]);
	mpz_mod_vec(scratch, scratch, mod, st->n[2]);
	polxvec_frommpzvec(st->cnst[i].phi[2], scratch, st->n[2]);

        // phi_3^{(i)} = -\gamma^{(i)} - \psi_i + C1[3]^t \epsilon
        // corresponds to wt.s[3] = c = Cw
	mpz_neg_vec(scratch, chall.round2.gamma, st->n[3]);
	mpz_sub_vec(scratch, scratch, chall.round1.psi, st->n[3]);
	mpz_mod_vec(scratch, scratch, mod, st->n[3]);
	polxvec_frommpzvec(st->cnst[i].phi[3], scratch, st->n[3]);


        // for all j \in [ELL] \phi_{4+j}^((i)} = C2[j]^t \zeta^{(i)} - delta_j
        // corresponds to wt.s[4+i] = d_i = \psi_i \circ a = \psi_i \circ Aw
        for (size_t j = 0; j != ELL; j++) {
	    mpz_neg_vec(scratch, chall.round2.deltas[j], st->n[4+j]);
	    mpz_mod_vec(scratch, scratch, mod, st->n[4+j]);
	    polxvec_frommpzvec(st->cnst[i].phi[4+j], scratch, st->n[4+j]);
        }


        // set st->cnst[i].b to whatever it actually evaluates to
        //sparsecnst_eval(st->cnst[i].b, &st->cnst[i], sx, &wt);

        // The verifier will check the following: (st->cnst[i].b - <commitment1, \epsilon^(i)> - <commitment2, \zeta^(i)>)(2) mod 2**N + 1 == 0
        // (and that st->cnst[i].b is computed correctly, via chihuahua/labrador)
        //polxvec_sprod(&g, commitments, epsilonx, m);
        //polxvec_sprod_add(&g, commitment2, zetax, md);
        //polx_sub(&g, st->cnst[i].b, &g);
    }
    free_mpz_array(scratch, MAX(st->n[0],st->n[1]));
    
}

void f_comm(prncplstmnt *st, challenge const *challenges, polx *commitments, polx const *const *C1, polx const *const *C2, size_t m, size_t md)
{
    for (size_t i = 0; i != ELL; i++) {
	for (size_t j = 0; j != 4; j++) {
	    polx_matmul_trans_add(st->cnst[i].phi[j], C1[j], challenges[i].round2.epsilon, m, st->n[j]);
	}
	for (size_t j = 0; j != ELL; j++) {
	    polx_matmul_trans_add(st->cnst[i].phi[4+j], C2[j], challenges[i].round2.zeta, md, st->n[j]);
	}
	polxvec_sprod_add(st->cnst[i].b, commitments, challenges[i].round2.epsilon, m);
	polxvec_sprod_add(st->cnst[i].b, commitments + m, challenges[i].round2.zeta, md);
    }
}

int64_t zz_toint64(zz const *a) {
    int64_t r = a->limbs[L-1];
    for (size_t i = 1; i != L; i++) {
	r <<= 14;
	r += a->limbs[L-1-i];
    }
    return r;

}

void polxvec_bin_decompose(poly *r, polx const *a, size_t len, int64_t nudge)
{

    uint64_t width = significant_bits((uint64_t) nudge) + 1; 
    assert(width < LOGQ - 1);
    polz nudge_polz;
    zz nudgezz;
    zz_fromint64(&nudgezz, nudge);
    for (size_t i = 0; i != N; i++) {
	polz_setcoeff(&nudge_polz, &nudgezz, i);
    }
    for (size_t i = 0; i != len; i++) {
	polz z;
	polz_frompolx(&z, a+i);
	polz_center(&z);
	polz_add(&z, &z, &nudge_polz);
	for (size_t j = 0; j != N; j++) {
	    zz coeffzz;
	    polz_getcoeff(&coeffzz, &z, j);
	    int64_t coeff = zz_toint64(&coeffzz);
	    assert(coeff > 0);
	    for (size_t k = 0; k != width; k++) {
		r[i*width+k].vec->c[j] =  coeff & 1;
		coeff >>= 1;
	    }
	}
    }
}

// sets the ith m-block of r to s*(1,2,4, ..., 2**(m-1))
void cnst_gadget_vec(polx *r, size_t m, size_t i, int64_t s)
{
    int64_t coeffs[N] = {};
    coeffs[0] = 1;
    for (size_t j = i*m; j != (i+1)*m; j++) {
	coeffs[0] *= s;
	polxvec_fromint64vec(r + j, 1, 1, coeffs);
	coeffs[0] = (coeffs[0]/s) << 1;
    }
}

void f_g_decomp(prncplstmnt *st, int64_t nudge, size_t width)
{
    int64_t nudge_coeffs[N];
    for (size_t i = 0; i != N; i++) {
        nudge_coeffs[i] = -nudge;
    }
    polx nudge_polx;
    polxvec_fromint64vec(&nudge_polx, 1, 1, nudge_coeffs);

    for (size_t i = 0; i != ELL; i++) {
        cnst_gadget_vec(st->cnst[i].phi[4+ELL], width, i, -1);
        polx_add(st->cnst[i].b, st->cnst[i].b, &nudge_polx);
    }
}



// generate proof for Aw \circ Bw = Cw (mod 'mod')
// C1 and C2 are commitment matrices.
// A,B,C have dimension k x n
// m, md are #rows of C1 and C2 respectively
void r1cs_reduction(mpz_sparsemat const *A, mpz_sparsemat const *B, mpz_sparsemat const *C, polx const *const *C1, polx const *const *C2, mpz_t *w, mpz_t const mod, uint8_t *hashstate, size_t k, size_t n, size_t m, size_t md)
{

    uint64_t gbits_width = significant_bits(GNORM(k,n))+1;

    // compute Aw Bw Cw, store contiguously
    mpz_t *mat_prods = new_mpz_array(3*k);
    sparsematmul(mat_prods, A, w, k, mod);
    sparsematmul(mat_prods + k, B, w, k, mod);
    sparsematmul(mat_prods + 2*k, C, w, k, mod);

    // coefficient vectors for w||Aw||Bw||Cw
    // TODO: include binary check in labrador constraints
    int64_t *encoded = _malloc((n+3*k) * N * sizeof *encoded);
    lift_to_coeffs_vector(encoded, w, n);
    for (size_t i = 0; i != 3; i++) {
        lift_to_coeffs_vector(encoded + (n + i*k) * N, mat_prods + i*k, k);
    }

    size_t wit_vecs_size = (ELL+3)*k + n + ELL*gbits_width;

    // poly representations of w||Aw||Bw||Cw
    // buffer has enough space for the rest of the witnesses computed later
    poly *wit_vecs = _aligned_alloc(64, wit_vecs_size * sizeof *wit_vecs);
    polyvec_fromint64vec(wit_vecs, n+3*k, 1, encoded);
    free(encoded);

    // polx version of wit_vecs, don't need polx versions for decomposed commitments
    polx *wit_vecsx = _aligned_alloc(64, wit_vecs_size * sizeof *wit_vecsx);
    polxvec_frompolyvec(wit_vecsx, wit_vecs, 3*k+n);

    // First commitment
    // TODO: include commitment check in labrador constraints
    polx *commitments = _aligned_alloc(64, (m + md) * sizeof *commitments);

    polx_matmul(commitments, C1[0], wit_vecsx, m, n);

    for (size_t i = 1; i != 4; i++) {
        polx_matmul_add(commitments, C1[i], wit_vecsx + n + ((i-1)*k), m, k);
    }

    // TODO also hash public params into hashstate at somepoint
    challenge *challenges = new_challenge_array(ELL, k, m, md);
    get_round1_challenges(challenges, hashstate, commitments, m, ELL);

    // ds[i] = psi[i] * Aw[i]
    mpz_t *ds = new_mpz_array(ELL*k);
    for (size_t i = 0; i != ELL; i++) {
	mpz_mul_vec(ds + i*k, challenges[i].round1.psi, mat_prods, k);
	mpz_mod_vec(ds + i*k, ds + i*k, mod, k);
    }
    free_mpz_array(mat_prods, 3*k);

    // coeff vectors of ds[i]
    int64_t *d_coeffs = _malloc(ELL*k*N*sizeof *d_coeffs);
    lift_to_coeffs_vector(d_coeffs, ds, k*ELL);
    free_mpz_array(ds, ELL*k);
    // poly representation of ds
    poly *wit_vecs2 = wit_vecs + (3*k+n);
    polyvec_fromint64vec(wit_vecs2, ELL*k, 1, d_coeffs);
    free(d_coeffs);

    // polx representatoin of ds
    polx *wit_vecsx2 = wit_vecsx + 3*k+n;
    polxvec_frompolyvec(wit_vecsx2, wit_vecs2, ELL*k);

    // second commitment
    polx *commitment2 = commitments + m;
    polxvec_setzero(commitment2, md);
    for (size_t i = 0; i != ELL; i++) {
        polx_matmul_add(commitment2, C2[i], wit_vecsx2 + i*k, md, k);
    }

    get_round2_challenges(challenges, hashstate, commitment2, md, ELL);

    // BEGIN constructing stuff for chihuahua call

    // num witness vecs
    size_t r = 5 + ELL;

    size_t wit_lens[r];

    // first witness (w) is length n, rest are k
    wit_lens[0] = n;
    for (size_t i = 1; i != 4+ELL; i++) {
        wit_lens[i] = k;
    }
    wit_lens[4+ELL] = gbits_width * ELL;

    witness wt = {};
    init_witness_raw_buf(&wt, r, wit_lens, wit_vecs);
    wit_vecs = NULL;


    // polx version of wt.s already have this as wit_vecsx, but need it in 2d array form
    polx *sx[r];

    sx[0] = wit_vecsx;
    wt.normsq[0] = polyvec_sprodz(wt.s[0], wt.s[0], wit_lens[0]);
    for (size_t i = 1; i != r; i++) {
        wt.normsq[i] = polyvec_sprodz(wt.s[i], wt.s[i], wit_lens[i]);
        sx[i] = sx[i-1] + wit_lens[i-1];
    }
    //uint64_t betasq = 128. / 30. * ((3+ELL)*k+n)*N/2;

    size_t idx[r];
    for (size_t i = 0; i != r; i++) {
	idx[i] = i;
    }

    prncplstmnt st = {};

    uint64_t betasq = beta_squared(k,n);

    init_prncplstmnt_raw(&st, r, wit_lens, betasq, ELL, 1);
    for (size_t i = 0; i != ELL; i++) {
        init_sparsecnst_raw(st.cnst + i, r, r, idx, wit_lens, 1, true, false);
	polxvec_setzero(st.cnst[i].b, 1);
    }

    f_r1cs(&st, A, B, C, challenges, mod);

    mpz_t eval;
    mpz_init(eval);
    polx *gs = _aligned_alloc(64, ELL * sizeof *gs);
    for (size_t i = 0; i != ELL; i++) {
        sparsecnst_eval(gs + i, &st.cnst[i], sx, &wt);
	polx_eval(eval, gs + i, 2, mod);
	assert(mpz_sgn(eval) == 0);
    }
    mpz_clear(eval);

    polxvec_bin_decompose(wt.s[4+ELL], gs, ELL, GNORM(k,n));
    polxvec_frompolyvec(sx[4+ELL], wt.s[4+ELL], ELL*gbits_width);
    
    // BEGIN ROUND3


    f_g_decomp(&st, GNORM(k,n), gbits_width);
    f_comm(&st, challenges, commitments, C1, C2, m, md);


    memcpy(st.h, hashstate, 16);
    for (size_t i = 0; i != ELL; i++) {
	sparsecnst_hash(st.h, st.cnst + i, r, wit_lens, 1);
    }

    print_prncplstmnt_pp(&st);
    // should pass trivially per above
    assert(principle_verify(&st, &wt) == 0);


    composite cproof = {};
    assert(composite_prove_principle(&cproof, &st, &wt) == 0);
    printf("%d\n", composite_verify_principle(&cproof, &st));

    free(wit_vecsx);
    free(commitments);
    free_challenge_array(challenges, ELL);
    //free(epsilonx);
    //free(zetax);
    free_witness(&wt); // frees wit_vecs
    free_prncplstmnt(&st);
    free(gs);
}


int main(void)
{
    // placeholders
    size_t k = 1000;
    size_t n = 1000;
    // todo: replace these with approproate values
    size_t m = 3;
    size_t md = 3;


    gmp_randstate_t grand;
    gmp_randinit_default(grand);
    mpz_t *w = new_mpz_array(n);
    for (size_t i = 0; i != n; i++) {
	mpz_urandomb(w[i], grand, N);
    }

    mpz_sparsemat A, B, C;
    init_mpz_sparsemat(&A, n);
    init_mpz_sparsemat(&B, n);
    init_mpz_sparsemat(&C, n);

    for (size_t i = 0; i != n; i++) {
        add_entry_ui(&A, i, i, 1);
        add_entry_ui(&B, i, i, 1);
        add_entry(&C, i, i, w[i]);
    }


    mpz_t mod;
    mpz_init(mod);
    mpz_set_str(mod, MOD_STR, 10);


    polx *comm = _aligned_alloc(64, ((3*k+n)*m + md * ELL*k) * sizeof *comm);

    __attribute__((aligned(16)))
    unsigned char seed[16] = {150, 98, 20, 81, 126, 151, 66, 43, 68, 235, 210, 118, 199, 77, 163, 30};
    int64_t nonce = 0;
    polxvec_almostuniform(comm, (3*k+n)*m + md * ELL * k, seed, nonce);
    polx const *C1[4];
    C1[0] = comm;
    C1[1] = comm + n*m;
    for (size_t i = 2; i != 4; i++) {
        C1[i] = C1[i-1] + k*m;
    }
    polx const *C2[ELL];
    C2[0] = comm + (3*k+n)*m;
    for (size_t i = 1; i != ELL; i++) {
        C2[i] = C2[i-1] + k*md;
    }

    r1cs_reduction(&A, &B, &C, C1, C2, w, mod, seed, k, n, m, md);
    free_mpz_sparsemat(&A);
    free_mpz_sparsemat(&B);
    free_mpz_sparsemat(&C);
    free(comm);
    mpz_clear(mod);
    free_mpz_array(w, n);

}

