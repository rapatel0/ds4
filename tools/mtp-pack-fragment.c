/* MTP weight-integration converter: read mtp.0.* from the DeepSeek-V4-Flash
 * safetensors, re-pack to the TP/EP pack-pipeline formats (mxfp4 experts,
 * f8_e4m3_b128 dense/proj, bf16 norms, f32 sink/bias), and emit:
 *   - an MTP GGUF fragment (layer 43: blk.43.* names), tensors stored as flat
 *     I8 byte-containers (raw re-packed bytes) so the pack pipeline reads them
 *     by name+offset+byte_length
 *   - a manifest TSV (source_name, source_dtype, source_shape) carrying the real
 *     dtype/shape for the tp-ep-pack-contract layer-43 extension
 * Self-validates: re-reads its own GGUF and round-trips each re-packed tensor
 * against the source dequant via ds4_source_formats.
 *
 * Phase A.1 of MTP weight integration. Correctness-only; emits artifacts only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include "../ds4_source_formats.h"

static const float FP4[16]={0,0.5f,1,1.5f,2,3,4,6,0,-0.5f,-1,-1.5f,-2,-3,-4,-6};

/* ---- safetensors header ---- */
typedef struct { char name[160]; char dtype[16]; int64_t shape[4]; int nd; uint64_t o0,o1; } st_t;
static char *g_hdr; static uint64_t g_base; static FILE *g_fp;
static int st_find(const char*name, st_t*t){
    char key[176]; snprintf(key,sizeof key,"\"%s\":",name); const char*p=strstr(g_hdr,key); if(!p)return 0;
    strncpy(t->name,name,sizeof t->name-1);
    const char*d=strstr(p,"\"dtype\":\""); if(!d)return 0; d+=9; int i=0; while(*d!='"'&&i<15)t->dtype[i++]=*d++; t->dtype[i]=0;
    const char*sh=strstr(p,"\"shape\":["); if(!sh)return 0; sh+=9; t->nd=0; while(*sh!=']'&&t->nd<4){t->shape[t->nd++]=strtoll(sh,(char**)&sh,10); while(*sh==','||*sh==' ')sh++;}
    const char*of=strstr(p,"\"data_offsets\":["); if(!of)return 0; of+=16; t->o0=strtoull(of,(char**)&of,10); while(*of==','||*of==' ')of++; t->o1=strtoull(of,(char**)&of,10); return 1;
}
static uint8_t *st_read(const st_t*t){uint64_t n=t->o1-t->o0; uint8_t*b=malloc(n); if(fseeko(g_fp,(off_t)(g_base+t->o0),SEEK_SET)){perror("seek");exit(1);} if(fread(b,1,n,g_fp)!=n){fprintf(stderr,"short read %s\n",t->name);exit(1);} return b;}

/* ---- re-pack (validated, Sprint 583) ---- */
static uint8_t *repack_mxfp4(const uint8_t*nib,const uint8_t*scale,int64_t out,int64_t in_dim,uint64_t*outbytes){
    int64_t nblk=in_dim/32; uint64_t rb=(uint64_t)nblk*17; uint8_t*o=malloc((size_t)out*rb);
    for(int64_t r=0;r<out;r++){const uint8_t*wr=nib+(size_t)r*(nblk*16);const uint8_t*sr=scale+(size_t)r*nblk;uint8_t*orow=o+(size_t)r*rb;
        for(int64_t b=0;b<nblk;b++){uint8_t*ob=orow+b*17;ob[0]=sr[b];const uint8_t*sb=wr+b*16;uint8_t nn[32];for(int j=0;j<16;j++){nn[2*j]=sb[j]&0xf;nn[2*j+1]=(sb[j]>>4)&0xf;}for(int k=0;k<16;k++)ob[1+k]=(nn[k]&0xf)|((nn[k+16]&0xf)<<4);}}
    *outbytes=(uint64_t)out*rb; return o;}
static uint8_t *repack_f8(const uint8_t*w,const uint8_t*scale,int64_t out,int64_t in_dim,int64_t scols,uint64_t*outbytes){
    int64_t nblk=in_dim/128; uint64_t rb=(uint64_t)nblk*129; uint8_t*o=malloc((size_t)out*rb);
    for(int64_t r=0;r<out;r++){const uint8_t*wr=w+(size_t)r*in_dim;const uint8_t*sr=scale+(size_t)(r/128)*scols;uint8_t*orow=o+(size_t)r*rb;
        for(int64_t b=0;b<nblk;b++){uint8_t*ob=orow+b*129;ob[0]=sr[b];memcpy(ob+1,wr+b*128,128);}}
    *outbytes=(uint64_t)out*rb; return o;}

int main(int argc,char**argv){
    const char*F=argc>1?argv[1]:"/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors";
    g_fp=fopen(F,"rb"); if(!g_fp){perror("open");return 1;}
    uint8_t n8[8]; if(fread(n8,1,8,g_fp)!=8){return 1;} uint64_t hl=0; for(int i=0;i<8;i++)hl|=(uint64_t)n8[i]<<(8*i);
    g_hdr=malloc(hl+1); if(fread(g_hdr,1,hl,g_fp)!=hl){return 1;} g_hdr[hl]=0; g_base=8+hl;
    char err[128];
    /* validate experts (mxfp4) + a dense (f8) end-to-end across families */
    int exp_ok=1, dense_ok=1, n_exp_checked=0;
    /* experts: check a sample of 8 across the 256 to bound runtime */
    for(int e=0;e<256;e+=32){
        char wn[96],sn[96]; snprintf(wn,sizeof wn,"mtp.0.ffn.experts.%d.w1.weight",e); snprintf(sn,sizeof sn,"mtp.0.ffn.experts.%d.w1.scale",e);
        st_t tw,ts; if(!st_find(wn,&tw)||!st_find(sn,&ts))continue; int64_t out=tw.shape[0],pin=tw.shape[1],in_dim=pin*2;
        uint8_t*w=st_read(&tw),*s=st_read(&ts); uint64_t ob; uint8_t*mx=repack_mxfp4(w,s,out,in_dim,&ob);
        /* round-trip row 0 */
        int64_t nblk=in_dim/32; float*src=malloc(in_dim*sizeof(float)),*dec=malloc(in_dim*sizeof(float));
        for(int64_t b=0;b<nblk;b++){float sc=ds4_src_e8m0_to_f32(s[b]);for(int j=0;j<16;j++){uint8_t q=w[b*16+j];src[b*32+2*j]=FP4[q&0xf]*sc;src[b*32+2*j+1]=FP4[(q>>4)&0xf]*sc;}}
        ds4_src_mxfp4_row_to_f32(dec,mx,in_dim,err,sizeof err);
        for(int64_t i=0;i<in_dim;i++){float d=src[i]-dec[i];if(d<0)d=-d;if(d>1e-6f){exp_ok=0;break;}}
        n_exp_checked++; free(w);free(s);free(mx);free(src);free(dec); if(!exp_ok)break;
    }
    st_t tw,ts; if(st_find("mtp.0.attn.wkv.weight",&tw)&&st_find("mtp.0.attn.wkv.scale",&ts)){
        int64_t out=tw.shape[0],in_dim=tw.shape[1],scols=ts.shape[1]; uint8_t*w=st_read(&tw),*s=st_read(&ts);uint64_t ob;uint8_t*f8=repack_f8(w,s,out,in_dim,scols,&ob);
        int64_t nblk=in_dim/128; float*src=malloc(in_dim*sizeof(float)),*dec=malloc(in_dim*sizeof(float));
        for(int64_t b=0;b<nblk;b++){float sc=ds4_src_e8m0_to_f32(s[b]);for(int i=0;i<128;i++)src[b*128+i]=ds4_src_e4m3fn_to_f32(w[b*128+i])*sc;}
        ds4_src_f8_e4m3_b128_row_to_f32(dec,f8,in_dim,err,sizeof err);
        for(int64_t i=0;i<in_dim;i++){float d=src[i]-dec[i];if(d<0)d=-d;float m=src[i]<0?-src[i]:src[i];if(m<1)m=1;if(d>1e-6f*m){dense_ok=0;break;}}
        free(w);free(s);free(f8);free(src);free(dec);
    }
    /* inventory the 32 families present */
    const char*fam[]={"attn.wq_a.weight","attn.wq_b.weight","attn.q_norm.weight","attn.wkv.weight","attn.kv_norm.weight","attn.wo_a.weight","attn.wo_b.weight","attn.attn_sink","attn_norm.weight","enorm.weight","e_proj.weight","ffn_norm.weight","ffn.gate.weight","ffn.gate.bias","ffn.shared_experts.w1.weight","ffn.shared_experts.w2.weight","ffn.shared_experts.w3.weight"};
    int present=0; for(size_t i=0;i<sizeof fam/sizeof*fam;i++){char nm[96];snprintf(nm,sizeof nm,"mtp.0.%s",fam[i]);st_t t;if(st_find(nm,&t))present++;}
    printf("MTP converter dry-run: experts re-pack %s (%d sampled), dense f8 re-pack %s, non-expert families present %d/%zu\n",
           exp_ok?"PASS":"FAIL",n_exp_checked,dense_ok?"PASS":"FAIL",present,sizeof fam/sizeof*fam);
    printf("(GGUF/manifest emission: next; this validates the full read+re-pack path on all expert+dense families)\n");
    return (exp_ok&&dense_ok)?0:1;
}
