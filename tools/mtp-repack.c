/* MTP weight-integration: source (deepseek4 safetensors) -> pack-pipeline
 * format re-pack core + round-trip self-test. Lossless: same fp4/e4m3 values
 * and e8m0 scales, only byte/nibble layout changes.
 *
 *   experts:  fp4 interleaved (elem 2j=low(src[j]), 2j+1=high(src[j])), one
 *             e8m0 scale per 32-block  ->  mxfp4 (17B/block: [e8m0][16 bytes],
 *             low nibbles -> elems 0..15, high -> 16..31)
 *   dense:    F8_E4M3 [out,in] + F8_E8M0 [out/128,in/128]  ->  f8_e4m3_b128
 *             (129B/block: [e8m0][128 e4m3]) per-row 128-blocks
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../ds4_source_formats.h"

/* one expert weight row [in_dim] from interleaved-fp4 src (in_dim/2 bytes) +
 * e8m0 scales (in_dim/32 bytes) -> mxfp4 row bytes (nblocks*17). */
static void repack_fp4_row_to_mxfp4(uint8_t *dst, const uint8_t *src_nib,
                                    const uint8_t *src_scale, uint64_t in_dim) {
    const uint64_t nblk = in_dim / 32;
    for (uint64_t b = 0; b < nblk; b++) {
        uint8_t *o = dst + b * 17;
        o[0] = src_scale[b];                 /* same e8m0 scale byte */
        /* source block: 16 bytes, elem 2j=low(src[j]), 2j+1=high(src[j]) */
        const uint8_t *sb = src_nib + b * 16;
        uint8_t nib[32];
        for (int j = 0; j < 16; j++) { nib[2*j] = sb[j] & 0x0f; nib[2*j+1] = (sb[j] >> 4) & 0x0f; }
        /* target: byte k packs low=elem k (k<16), high=elem k+16 */
        for (int k = 0; k < 16; k++) o[1+k] = (nib[k] & 0x0f) | ((nib[k+16] & 0x0f) << 4);
    }
}

/* one dense weight row [in_dim] of F8_E4M3 + that row-block's e8m0 scales
 * (in_dim/128 bytes) -> f8_e4m3_b128 row bytes (nblocks*129). */
static void repack_f8_row_to_b128(uint8_t *dst, const uint8_t *src_e4m3,
                                  const uint8_t *row_scale, uint64_t in_dim) {
    const uint64_t nblk = in_dim / 128;
    for (uint64_t b = 0; b < nblk; b++) {
        uint8_t *o = dst + b * 129;
        o[0] = row_scale[b];
        memcpy(o + 1, src_e4m3 + b * 128, 128);
    }
}

static int approx(float a, float b, float tol) { float d=a-b; if(d<0)d=-d; float m=a<0?-a:a; if(m<1)m=1; return d <= tol*m; }

int main(void) {
    char err[128];
    /* --- mxfp4 round-trip: random fp4 block, re-pack, decode both layouts --- */
    const uint64_t in_dim = 64; /* 2 blocks */
    uint8_t src_nib[32], src_scale[2];
    for (int i=0;i<32;i++) src_nib[i]=(uint8_t)(rand()&0xff);
    src_scale[0]=127; src_scale[1]=128; /* e8m0: 2^0, 2^1 */
    /* decode source the deepseek4 way */
    static const float fp4[16]={0,0.5f,1,1.5f,2,3,4,6,0,-0.5f,-1,-1.5f,-2,-3,-4,-6};
    float src_f32[64];
    for (uint64_t b=0;b<2;b++){ float s=ds4_src_e8m0_to_f32(src_scale[b]); for(int j=0;j<16;j++){uint8_t q=src_nib[b*16+j]; src_f32[b*32+2*j]=fp4[q&0xf]*s; src_f32[b*32+2*j+1]=fp4[(q>>4)&0xf]*s;} }
    uint8_t mx[2*17]; repack_fp4_row_to_mxfp4(mx, src_nib, src_scale, in_dim);
    float dec[64];
    if (ds4_src_mxfp4_row_to_f32(dec, mx, in_dim, err, sizeof err)!=0){printf("mxfp4 decode err: %s\n",err);return 1;}
    int bad=0; for(int i=0;i<64;i++) if(!approx(src_f32[i],dec[i],1e-6f)){bad++; if(bad<=4)printf("  mxfp4 mismatch elem %d: src %.4f dec %.4f\n",i,src_f32[i],dec[i]);}
    printf("mxfp4 re-pack round-trip: %s (%d/64 mismatch)\n", bad?"FAIL":"PASS", bad);

    /* --- f8_e4m3_b128 round-trip: 1 row, in_dim=256 (2 blocks) --- */
    const uint64_t fin=256; uint8_t e4m3[256], rscale[2]; for(int i=0;i<256;i++)e4m3[i]=(uint8_t)(rand()%0x7f); rscale[0]=127; rscale[1]=126;
    float fsrc[256]; for(uint64_t b=0;b<2;b++){float s=ds4_src_e8m0_to_f32(rscale[b]); for(int i=0;i<128;i++) fsrc[b*128+i]=ds4_src_e4m3fn_to_f32(e4m3[b*128+i])*s;}
    uint8_t f8[2*129]; repack_f8_row_to_b128(f8,e4m3,rscale,fin);
    float fdec[256]; if(ds4_src_f8_e4m3_b128_row_to_f32(fdec,f8,fin,err,sizeof err)!=0){printf("f8 decode err: %s\n",err);return 1;}
    int fbad=0; for(int i=0;i<256;i++) if(!approx(fsrc[i],fdec[i],1e-6f))fbad++;
    printf("f8_e4m3_b128 re-pack round-trip: %s (%d/256 mismatch)\n", fbad?"FAIL":"PASS", fbad);
    return (bad||fbad)?1:0;
}
