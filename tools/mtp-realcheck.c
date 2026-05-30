/* Validate the MTP re-pack core on REAL safetensors weights: read an expert
 * (packed-fp4+e8m0) and a dense (f8_e4m3+e8m0) tensor + their scales, dequant
 * (deepseek4 convention), re-pack to mxfp4 / f8_e4m3_b128, decode via
 * ds4_source_formats, and confirm the round-trip matches on real data. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include "../ds4_source_formats.h"

static char *slurp_header(FILE *fp, uint64_t *hlen){
    uint8_t n8[8]; if(fread(n8,1,8,fp)!=8){fprintf(stderr,"short hdr len\n");exit(1);}
    uint64_t n=0; for(int i=0;i<8;i++) n|=(uint64_t)n8[i]<<(8*i);
    char *h=malloc(n+1); if(fread(h,1,n,fp)!=n){fprintf(stderr,"short hdr\n");exit(1);} h[n]=0; *hlen=n; return h;
}
/* crude JSON field extract for "name": {... "dtype":"X","data_offsets":[a,b] ...} */
static int find_tensor(const char*h,const char*name,char*dtype,uint64_t*o0,uint64_t*o1){
    char key[256]; snprintf(key,sizeof key,"\"%s\":",name); const char*p=strstr(h,key); if(!p)return 0;
    const char*d=strstr(p,"\"dtype\":\""); if(!d)return 0; d+=9; int i=0; while(*d!='"'&&i<31)dtype[i++]=*d++; dtype[i]=0;
    const char*off=strstr(p,"\"data_offsets\":["); if(!off)return 0; off+=16; *o0=strtoull(off,(char**)&off,10); while(*off==','||*off==' ')off++; *o1=strtoull(off,(char**)&off,10); return 1;
}
static uint8_t *read_at(FILE*fp,uint64_t base,uint64_t o0,uint64_t o1){uint64_t n=o1-o0; uint8_t*b=malloc(n); if(fseeko(fp,(off_t)(base+o0),SEEK_SET)){perror("seek");exit(1);} if(fread(b,1,n,fp)!=n){fprintf(stderr,"short tensor read\n");exit(1);} return b;}
static const float FP4[16]={0,0.5f,1,1.5f,2,3,4,6,0,-0.5f,-1,-1.5f,-2,-3,-4,-6};

int main(int argc,char**argv){
    const char*F=argc>1?argv[1]:"/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors";
    FILE*fp=fopen(F,"rb"); if(!fp){perror("open");return 1;} uint64_t hlen; char*h=slurp_header(fp,&hlen); uint64_t base=8+hlen; char err[128];

    /* --- expert: mtp.0.ffn.experts.0.w1 (I8 packed-fp4) + .scale (F8_E8M0) --- */
    char dt[32]; uint64_t a,b;
    if(find_tensor(h,"mtp.0.ffn.experts.0.w1.weight",dt,&a,&b)){
        /* shape [out, packed_in]; read row 0 only for a quick check */
        /* parse shape */
        char key[128]; snprintf(key,sizeof key,"\"%s\":","mtp.0.ffn.experts.0.w1.weight"); const char*sp=strstr(strstr(h,key),"\"shape\":["); sp+=9;
        long outd=strtol(sp,(char**)&sp,10); while(*sp==','||*sp==' ')sp++; long pin=strtol(sp,(char**)&sp,10); long in_dim=pin*2;
        char sdt[32]; uint64_t sa,sb; find_tensor(h,"mtp.0.ffn.experts.0.w1.scale",sdt,&sa,&sb);
        long nblk=in_dim/32;
        uint8_t*wrow=read_at(fp,base,a,a+(uint64_t)pin);            /* row0 nibble bytes: pin = nblk*16 */
        uint8_t*srow=read_at(fp,base,sa,sa+(uint64_t)nblk);         /* row0 scales: nblk bytes */
        float*src=malloc((size_t)in_dim*sizeof(float));
        for(long bb=0;bb<nblk;bb++){float s=ds4_src_e8m0_to_f32(srow[bb]); for(int j=0;j<16;j++){uint8_t q=wrow[bb*16+j]; src[bb*32+2*j]=FP4[q&0xf]*s; src[bb*32+2*j+1]=FP4[(q>>4)&0xf]*s;}}
        /* re-pack row0 -> mxfp4 */
        uint8_t*mx=malloc((size_t)nblk*17);
        for(long bb=0;bb<nblk;bb++){uint8_t*o=mx+bb*17;o[0]=srow[bb];const uint8_t*sb2=wrow+bb*16;uint8_t nib[32];for(int j=0;j<16;j++){nib[2*j]=sb2[j]&0xf;nib[2*j+1]=(sb2[j]>>4)&0xf;}for(int k=0;k<16;k++)o[1+k]=(nib[k]&0xf)|((nib[k+16]&0xf)<<4);}
        float*dec=malloc((size_t)in_dim*sizeof(float));
        if(ds4_src_mxfp4_row_to_f32(dec,mx,(uint64_t)in_dim,err,sizeof err)){printf("expert decode err %s\n",err);return 1;}
        int bad=0; for(long i=0;i<in_dim;i++){float d=src[i]-dec[i];if(d<0)d=-d;if(d>1e-6f)bad++;}
        printf("REAL expert w1 row0 (dtype=%s in_dim=%ld nblk=%ld): mxfp4 round-trip %s (%d/%ld)\n",dt,in_dim,nblk,bad?"FAIL":"PASS",bad,in_dim);
    } else printf("expert tensor not found\n");

    /* --- dense: mtp.0.attn.wkv (F8_E4M3) + .scale (F8_E8M0) row0 --- */
    if(find_tensor(h,"mtp.0.attn.wkv.weight",dt,&a,&b)){
        char key[128]; snprintf(key,sizeof key,"\"%s\":","mtp.0.attn.wkv.weight"); const char*sp=strstr(strstr(h,key),"\"shape\":["); sp+=9;
        long outd=strtol(sp,(char**)&sp,10); while(*sp==','||*sp==' ')sp++; long ind=strtol(sp,(char**)&sp,10);
        char sdt[32]; uint64_t sa,sb; find_tensor(h,"mtp.0.attn.wkv.scale",sdt,&sa,&sb);
        long nblk=ind/128;
        uint8_t*wrow=read_at(fp,base,a,a+(uint64_t)ind);                 /* row0: ind e4m3 bytes */
        uint8_t*srow=read_at(fp,base,sa,sa+(uint64_t)nblk);              /* row0 (block 0 of 128): nblk col-block scales */
        float*src=malloc((size_t)ind*sizeof(float));
        for(long bb=0;bb<nblk;bb++){float s=ds4_src_e8m0_to_f32(srow[bb]);for(int i=0;i<128;i++)src[bb*128+i]=ds4_src_e4m3fn_to_f32(wrow[bb*128+i])*s;}
        uint8_t*f8=malloc((size_t)nblk*129); for(long bb=0;bb<nblk;bb++){uint8_t*o=f8+bb*129;o[0]=srow[bb];memcpy(o+1,wrow+bb*128,128);}
        float*dec=malloc((size_t)ind*sizeof(float));
        if(ds4_src_f8_e4m3_b128_row_to_f32(dec,f8,(uint64_t)ind,err,sizeof err)){printf("dense decode err %s\n",err);return 1;}
        int bad=0; for(long i=0;i<ind;i++){float d=src[i]-dec[i];if(d<0)d=-d;float m=src[i]<0?-src[i]:src[i];if(m<1)m=1;if(d>1e-6f*m)bad++;}
        printf("REAL dense wkv row0 (dtype=%s in_dim=%ld nblk=%ld): f8_e4m3_b128 round-trip %s (%d/%ld)\n",dt,ind,nblk,bad?"FAIL":"PASS",bad,ind);
    } else printf("dense tensor not found\n");
    return 0;
}
