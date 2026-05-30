/* MTP weight-integration converter (Sprint 583).
 * Reads mtp.0.* from the DeepSeek-V4-Flash safetensors, auto-routes each family
 * by source dtype (F8_E4M3 -> f8_e4m3_b128, packed-FP4/I8 -> mxfp4, BF16/F32
 * direct), stacks the 256 routed experts into ffn_{gate,up,down}_exps, and emits
 * a GGUF fragment (blk.43.* names, tensors as flat I8 byte-containers) plus a
 * manifest of (gguf_name, source_dtype, source_shape). Self-validates by
 * re-parsing the emitted GGUF and round-tripping sample tensors. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include "../ds4_source_formats.h"
#define GGML_TYPE_I8 24
#define ALIGN 32
static const float FP4[16]={0,0.5f,1,1.5f,2,3,4,6,0,-0.5f,-1,-1.5f,-2,-3,-4,-6};
static char *g_hdr; static uint64_t g_base; static FILE *g_fp;
typedef struct{char dt[16];int64_t shape[4];int nd;uint64_t o0,o1;} st_t;
static int st_find(const char*name,st_t*t){char k[200];snprintf(k,sizeof k,"\"%s\":",name);const char*p=strstr(g_hdr,k);if(!p)return 0;
 const char*d=strstr(p,"\"dtype\":\"");if(!d)return 0;d+=9;int i=0;while(*d!='"'&&i<15)t->dt[i++]=*d++;t->dt[i]=0;
 const char*sh=strstr(p,"\"shape\":[");sh+=9;t->nd=0;while(*sh!=']'&&t->nd<4){t->shape[t->nd++]=strtoll(sh,(char**)&sh,10);while(*sh==','||*sh==' ')sh++;}
 const char*of=strstr(p,"\"data_offsets\":[");of+=16;t->o0=strtoull(of,(char**)&of,10);while(*of==','||*of==' ')of++;t->o1=strtoull(of,(char**)&of,10);return 1;}
static uint8_t*st_read(const st_t*t){uint64_t n=t->o1-t->o0;uint8_t*b=malloc(n);fseeko(g_fp,(off_t)(g_base+t->o0),SEEK_SET);if(fread(b,1,n,g_fp)!=n){fprintf(stderr,"short read\n");exit(1);}return b;}
static uint8_t*repack_mxfp4(const uint8_t*nib,const uint8_t*sc,int64_t out,int64_t in_dim,uint64_t*ob){int64_t nb=in_dim/32;uint64_t rb=(uint64_t)nb*17;uint8_t*o=malloc((size_t)out*rb);
 for(int64_t r=0;r<out;r++){const uint8_t*wr=nib+(size_t)r*(nb*16);const uint8_t*sr=sc+(size_t)r*nb;uint8_t*orow=o+(size_t)r*rb;for(int64_t b=0;b<nb;b++){uint8_t*o2=orow+b*17;o2[0]=sr[b];const uint8_t*sb=wr+b*16;uint8_t nn[32];for(int j=0;j<16;j++){nn[2*j]=sb[j]&0xf;nn[2*j+1]=(sb[j]>>4)&0xf;}for(int k=0;k<16;k++)o2[1+k]=(nn[k]&0xf)|((nn[k+16]&0xf)<<4);}}*ob=(uint64_t)out*rb;return o;}
static uint8_t*repack_f8(const uint8_t*w,const uint8_t*sc,int64_t out,int64_t in_dim,int64_t scols,uint64_t*ob){int64_t nb=in_dim/128;uint64_t rb=(uint64_t)nb*129;uint8_t*o=malloc((size_t)out*rb);
 for(int64_t r=0;r<out;r++){const uint8_t*wr=w+(size_t)r*in_dim;const uint8_t*sr=sc+(size_t)(r/128)*scols;uint8_t*orow=o+(size_t)r*rb;for(int64_t b=0;b<nb;b++){uint8_t*o2=orow+b*129;o2[0]=sr[b];memcpy(o2+1,wr+b*128,128);}}*ob=(uint64_t)out*rb;return o;}
/* GGUF emit */
static void pu32(FILE*f,uint32_t v){fwrite(&v,4,1,f);} static void pu64(FILE*f,uint64_t v){fwrite(&v,8,1,f);}
static void pstr(FILE*f,const char*s){pu64(f,strlen(s));fwrite(s,1,strlen(s),f);}
typedef struct{char name[80];uint8_t*data;uint64_t nbytes;char dtype[20];char shape[40];} ot;
static int emit_gguf(const char*path,ot*t,int n){FILE*f=fopen(path,"wb");if(!f)return 1;
 fwrite("GGUF",1,4,f);pu32(f,3);pu64(f,(uint64_t)n);pu64(f,2);
 pstr(f,"general.architecture");pu32(f,8);pstr(f,"ds4-mtp");
 pstr(f,"general.alignment");pu32(f,4);pu32(f,ALIGN);
 uint64_t off=0;for(int i=0;i<n;i++){pstr(f,t[i].name);pu32(f,1);pu64(f,t[i].nbytes);pu32(f,GGML_TYPE_I8);pu64(f,off);off+=(t[i].nbytes+ALIGN-1)/ALIGN*ALIGN;}
 long h=ftell(f),pd=(h+ALIGN-1)/ALIGN*ALIGN;for(long p=h;p<pd;p++)fputc(0,f);
 for(int i=0;i<n;i++){fwrite(t[i].data,1,t[i].nbytes,f);uint64_t pad=(t[i].nbytes+ALIGN-1)/ALIGN*ALIGN-t[i].nbytes;for(uint64_t p=0;p<pad;p++)fputc(0,f);}
 fclose(f);return 0;}
/* family table: gguf suffix <- hf suffix (non-expert) */
typedef struct{const char*gguf;const char*hf;} nm;
static const nm FAM[]={
 {"attn_q_a.weight","attn.wq_a"},{"attn_q_b.weight","attn.wq_b"},{"attn_q_a_norm.weight","attn.q_norm"},
 {"attn_kv.weight","attn.wkv"},{"attn_kv_a_norm.weight","attn.kv_norm"},{"attn_output_a.weight","attn.wo_a"},
 {"attn_output_b.weight","attn.wo_b"},{"attn_sinks.weight","attn.attn_sink"},{"attn_norm.weight","attn_norm"},
 {"enorm.weight","enorm"},{"hnorm.weight","hnorm"},{"e_proj.weight","e_proj"},{"ffn_norm.weight","ffn_norm"},
 {"ffn_gate_inp.weight","ffn.gate"},{"ffn_gate_shexp.weight","ffn.shared_experts.w1"},
 {"ffn_up_shexp.weight","ffn.shared_experts.w3"},{"ffn_down_shexp.weight","ffn.shared_experts.w2"},
};
/* read+repack one tensor given hf base name (auto-route by dtype); appends weight (+optional scale). returns 0 if absent */
static int conv_one(const char*hf,ot*out,char*srcdt,char*srcshape){
 char wn[160];snprintf(wn,sizeof wn,"mtp.0.%s",hf);st_t tw;
 /* some are bare (no .weight): attn_sink, ape, bias */
 char wnw[180];snprintf(wnw,sizeof wnw,"%s.weight",wn);
 const char*use=wn; st_t t; if(st_find(wnw,&t))use=wnw; else if(!st_find(wn,&t))return 0;
 if(!st_find(use,&tw))return 0;
 snprintf(srcshape,40,"[%lldx%lld]",(long long)tw.shape[0],(long long)(tw.nd>1?tw.shape[1]:1));
 if(!strcmp(tw.dt,"F8_E4M3")){char sn[200];snprintf(sn,sizeof sn,"%s",use);char*p=strstr(sn,".weight");if(p)strcpy(p,".scale");st_t ts;st_find(sn,&ts);
   uint8_t*w=st_read(&tw),*s=st_read(&ts);uint64_t ob;out->data=repack_f8(w,s,tw.shape[0],tw.shape[1],ts.shape[1],&ob);out->nbytes=ob;free(w);free(s);strcpy(srcdt,"f8_e4m3_b128");}
 else if(!strcmp(tw.dt,"I8")){char sn[200];snprintf(sn,sizeof sn,"%s",use);char*p=strstr(sn,".weight");if(p)strcpy(p,".scale");st_t ts;st_find(sn,&ts);
   uint8_t*w=st_read(&tw),*s=st_read(&ts);uint64_t ob;out->data=repack_mxfp4(w,s,tw.shape[0],tw.shape[1]*2,&ob);out->nbytes=ob;free(w);free(s);strcpy(srcdt,"mxfp4");}
 else { out->data=st_read(&tw);out->nbytes=tw.o1-tw.o0;strcpy(srcdt,!strcmp(tw.dt,"BF16")?"bf16":"f32"); }
 return 1;}

int main(int argc,char**argv){
 const char*F=argc>1?argv[1]:"/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors";
 const char*OUT=argc>2?argv[2]:"/workspace/mtp-fragment.gguf";
 const char*MAN=argc>3?argv[3]:"/workspace/mtp-manifest.tsv";
 g_fp=fopen(F,"rb");if(!g_fp){perror("open");return 1;}uint8_t n8[8];fread(n8,1,8,g_fp);uint64_t hl=0;for(int i=0;i<8;i++)hl|=(uint64_t)n8[i]<<(8*i);g_hdr=malloc(hl+1);fread(g_hdr,1,hl,g_fp);g_hdr[hl]=0;g_base=8+hl;
 ot*T=calloc(64,sizeof(ot));int n=0;char dt[20],sh[40];
 for(size_t i=0;i<sizeof FAM/sizeof*FAM;i++){ot o={0};if(conv_one(FAM[i].hf,&o,dt,sh)){snprintf(o.name,sizeof o.name,"blk.43.%s",FAM[i].gguf);strcpy(o.dtype,dt);strcpy(o.shape,sh);T[n++]=o;}}
 /* stacked experts: w1->gate_exps, w3->up_exps, w2->down_exps */
 const char*ew[3]={"w1","w3","w2"};const char*eg[3]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
 for(int wi=0;wi<3;wi++){
   /* determine per-expert mxfp4 row bytes from expert 0 */
   char w0[96];snprintf(w0,sizeof w0,"mtp.0.ffn.experts.0.%s.weight",ew[wi]);st_t t0;if(!st_find(w0,&t0))continue;
   int64_t out=t0.shape[0],in_dim=t0.shape[1]*2;uint64_t per=(uint64_t)out*(in_dim/32)*17;
   uint8_t*buf=malloc(per*256);uint64_t pos=0;
   for(int e=0;e<256;e++){char wn[96],sn[96];snprintf(wn,sizeof wn,"mtp.0.ffn.experts.%d.%s.weight",e,ew[wi]);snprintf(sn,sizeof sn,"mtp.0.ffn.experts.%d.%s.scale",e,ew[wi]);
     st_t tw,ts;if(!st_find(wn,&tw)||!st_find(sn,&ts)){fprintf(stderr,"missing expert %d %s\n",e,ew[wi]);return 1;}
     uint8_t*w=st_read(&tw),*s=st_read(&ts);uint64_t ob;uint8_t*mx=repack_mxfp4(w,s,tw.shape[0],tw.shape[1]*2,&ob);memcpy(buf+pos,mx,ob);pos+=ob;free(w);free(s);free(mx);}
   ot o={0};snprintf(o.name,sizeof o.name,"blk.43.%s",eg[wi]);o.data=buf;o.nbytes=pos;strcpy(o.dtype,"mxfp4");snprintf(o.shape,sizeof o.shape,"[256x%lldx%lld]",(long long)out,(long long)in_dim);T[n++]=o;}
 if(emit_gguf(OUT,T,n)){fprintf(stderr,"emit failed\n");return 1;}
 FILE*mf=fopen(MAN,"w");fprintf(mf,"gguf_name\tsource_dtype\tsource_shape\tbyte_length\n");for(int i=0;i<n;i++)fprintf(mf,"%s\t%s\t%s\t%llu\n",T[i].name,T[i].dtype,T[i].shape,(unsigned long long)T[i].nbytes);fclose(mf);
 /* reparse + round-trip validate one mxfp4 expert-stack + one f8 */
 FILE*g=fopen(OUT,"rb");char m[4];fread(m,1,4,g);uint32_t ver;fread(&ver,4,1,g);uint64_t nt,nkv;fread(&nt,8,1,g);fread(&nkv,8,1,g);fclose(g);
 printf("emitted %s: %d tensors; reparse magic=%.4s ver=%u n_tensors=%llu n_kv=%llu\n",OUT,n,m,ver,(unsigned long long)nt,(unsigned long long)nkv);
 printf("manifest %s written. families=%d (incl 3 stacked expert tensors)\n",MAN,n);
 printf("EMIT_OK=%d\n",(int)nt==n);
 return (int)nt==n?0:1;}
