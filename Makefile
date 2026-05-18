CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
PACK_OBJS = ds4_pack.o
SOURCE_FORMAT_OBJS = ds4_source_formats.o
V100_CONTEXT_OBJS = ds4_v100_context.o $(PACK_OBJS)
V100_LAYER_STATE_OBJS = ds4_v100_layer_state.o $(V100_CONTEXT_OBJS) $(SOURCE_FORMAT_OBJS)
V100_LAYER_EXECUTE_OBJS = ds4_v100_layer_execute.o $(V100_LAYER_STATE_OBJS)
V100_SCHEDULER_OBJS = ds4_v100_scheduler.o $(V100_LAYER_EXECUTE_OBJS)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o $(PACK_OBJS) $(SOURCE_FORMAT_OBJS)
CPU_CORE_OBJS = ds4_cpu.o $(PACK_OBJS) $(SOURCE_FORMAT_OBJS)
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CORE_OBJS = ds4.o ds4_cuda.o $(PACK_OBJS) $(SOURCE_FORMAT_OBJS)
CPU_CORE_OBJS = ds4_cpu.o $(PACK_OBJS) $(SOURCE_FORMAT_OBJS)
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=

cuda-generic:
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval CUDA_ARCH="$(CUDA_ARCH)"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4.o: ds4.c ds4.h ds4_gpu.h ds4_pack.h ds4_source_formats.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_pack.o: ds4_pack.c ds4_pack.h
	$(CC) $(CFLAGS) -c -o $@ ds4_pack.c

ds4_source_formats.o: ds4_source_formats.c ds4_source_formats.h
	$(CC) $(CFLAGS) -c -o $@ ds4_source_formats.c

ds4_v100_context.o: ds4_v100_context.c ds4_v100_context.h ds4_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ ds4_v100_context.c

ds4_v100_layer_state.o: ds4_v100_layer_state.c ds4_v100_layer_state.h ds4_v100_context.h ds4_gpu.h ds4_source_formats.h
	$(CC) $(CFLAGS) -I. -c -o $@ ds4_v100_layer_state.c

ds4_v100_layer_execute.o: ds4_v100_layer_execute.c ds4_v100_layer_execute.h ds4_v100_layer_state.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ ds4_v100_layer_execute.c

ds4_v100_scheduler.o: ds4_v100_scheduler.c ds4_v100_scheduler.h ds4_v100_layer_execute.h ds4_v100_layer_state.h ds4_v100_context.h ds4_pack.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ ds4_v100_scheduler.c

ds4_v100_context_cuda.o: ds4_v100_context_cuda.cu ds4_v100_context.h
	$(NVCC) $(NVCCFLAGS) -I. -c -o $@ ds4_v100_context_cuda.cu

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tools/ds4-v100-plan: tools/ds4-v100-plan.c
	$(CC) $(CFLAGS) -D_FILE_OFFSET_BITS=64 -o $@ tools/ds4-v100-plan.c $(LDLIBS)

tools/ds4-v100-pack: tools/ds4-v100-pack.c
	$(CC) $(CFLAGS) -D_FILE_OFFSET_BITS=64 -o $@ tools/ds4-v100-pack.c $(LDLIBS)

tools/ds4-v100-residency-smoke.o: tools/ds4-v100-residency-smoke.c ds4_pack.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -D_FILE_OFFSET_BITS=64 -c -o $@ tools/ds4-v100-residency-smoke.c

tools/ds4-v100-context-smoke.o: tools/ds4-v100-context-smoke.c ds4_v100_context.h
	$(CC) $(CFLAGS) -I. -D_FILE_OFFSET_BITS=64 -c -o $@ tools/ds4-v100-context-smoke.c

tools/ds4-v100-layer-descriptor-gate.o: tools/ds4-v100-layer-descriptor-gate.c ds4_pack.h
	$(CC) $(CFLAGS) -I. -D_FILE_OFFSET_BITS=64 -c -o $@ tools/ds4-v100-layer-descriptor-gate.c

tools/ds4-source-oracle-vector.o: tools/ds4-source-oracle-vector.c ds4.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -D_FILE_OFFSET_BITS=64 -c -o $@ tools/ds4-source-oracle-vector.c

ifeq ($(UNAME_S),Darwin)
tools/ds4-v100-residency-smoke: tools/ds4-v100-residency-smoke.o ds4_pack.o ds4_gpu_arena_stub.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
else
tools/ds4-v100-residency-smoke: tools/ds4-v100-residency-smoke.o ds4_pack.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

tools/ds4-v100-context-smoke: tools/ds4-v100-context-smoke.o $(V100_CONTEXT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tools/ds4-v100-layer-descriptor-gate: tools/ds4-v100-layer-descriptor-gate.o ds4_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tools/ds4-source-oracle-vector: tools/ds4-source-oracle-vector.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_gpu_arena_stub.o: ds4_gpu_arena_stub.c ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_arena_stub.c

tests/pack_index_smoke.o: tests/pack_index_smoke.c ds4_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/pack_index_smoke.c

tests/pack_index_smoke: tests/pack_index_smoke.o ds4_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/gpu_arena_smoke.o: tests/gpu_arena_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/gpu_arena_smoke.c

tests/gpu_arena_smoke: tests/gpu_arena_smoke.o ds4_gpu_arena_stub.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/bf16_probe_smoke.o: tests/bf16_probe_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/bf16_probe_smoke.c

tests/bf16_probe_smoke: tests/bf16_probe_smoke.o ds4_gpu_arena_stub.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/v100_context_smoke.o: tests/v100_context_smoke.c ds4_v100_context.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/v100_context_smoke.c

tests/v100_context_smoke: tests/v100_context_smoke.o $(V100_CONTEXT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/v100_layer_binding_smoke.o: tests/v100_layer_binding_smoke.c ds4_v100_context.h ds4_source_formats.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/v100_layer_binding_smoke.c

tests/v100_layer_binding_smoke: tests/v100_layer_binding_smoke.o $(V100_CONTEXT_OBJS) ds4_source_formats.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/v100_layer_state_smoke.o: tests/v100_layer_state_smoke.c ds4_v100_layer_state.h ds4_v100_context.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/v100_layer_state_smoke.c

tests/v100_layer_state_smoke: tests/v100_layer_state_smoke.o $(V100_LAYER_STATE_OBJS) ds4_gpu_arena_stub.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/source_dtypes_smoke.o: tests/source_dtypes_smoke.c ds4_source_formats.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/source_dtypes_smoke.c

tests/source_dtypes_smoke: tests/source_dtypes_smoke.o ds4_source_formats.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

tests/cuda_bf16_probe.o: tests/cuda_bf16_probe.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_bf16_probe.c

tests/cuda_v100_context_smoke.o: tests/cuda_v100_context_smoke.c ds4_v100_context.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_v100_context_smoke.c

tests/cuda_source_dtypes_smoke.o: tests/cuda_source_dtypes_smoke.c ds4_gpu.h ds4_source_formats.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_source_dtypes_smoke.c

tests/cuda_v100_prefill_kv_smoke.o: tests/cuda_v100_prefill_kv_smoke.c ds4_gpu.h ds4_source_formats.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_v100_prefill_kv_smoke.c

tests/cuda_v100_compressor_bridge_smoke.o: tests/cuda_v100_compressor_bridge_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_v100_compressor_bridge_smoke.c

tests/cuda_v100_projection_attention_smoke.o: tests/cuda_v100_projection_attention_smoke.c ds4_gpu.h ds4_source_formats.h ds4_v100_context.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_v100_projection_attention_smoke.c

tests/cuda_v100_bounded_logits_smoke.o: tests/cuda_v100_bounded_logits_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_v100_bounded_logits_smoke.c

tests/cuda_v100_mxfp4_moe_smoke.o: tests/cuda_v100_mxfp4_moe_smoke.c ds4_gpu.h ds4_source_formats.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_v100_mxfp4_moe_smoke.c

tests/cuda_v100_descriptor_bound_ffn_smoke.o: tests/cuda_v100_descriptor_bound_ffn_smoke.c ds4_gpu.h ds4_source_formats.h ds4_v100_context.h ds4_v100_layer_state.h
	$(CC) $(CFLAGS) -I. -D_FILE_OFFSET_BITS=64 -c -o $@ tests/cuda_v100_descriptor_bound_ffn_smoke.c

tests/cuda_v100_descriptor_bound_attention_smoke.o: tests/cuda_v100_descriptor_bound_attention_smoke.c ds4_gpu.h ds4_source_formats.h ds4_v100_layer_state.h
	$(CC) $(CFLAGS) -I. -D_FILE_OFFSET_BITS=64 -c -o $@ tests/cuda_v100_descriptor_bound_attention_smoke.c

tests/cuda_v100_integrated_layer_smoke.o: tests/cuda_v100_integrated_layer_smoke.c ds4_gpu.h ds4_source_formats.h ds4_v100_layer_state.h ds4_v100_layer_execute.h
	$(CC) $(CFLAGS) -I. -D_FILE_OFFSET_BITS=64 -c -o $@ tests/cuda_v100_integrated_layer_smoke.c

tests/cuda_v100_stage_scheduler_smoke.o: tests/cuda_v100_stage_scheduler_smoke.c ds4_v100_scheduler.h ds4_v100_layer_execute.h ds4_v100_context.h
	$(CC) $(CFLAGS) -I. -D_FILE_OFFSET_BITS=64 -c -o $@ tests/cuda_v100_stage_scheduler_smoke.c

tests/cuda_hc_relay_smoke.o: tests/cuda_hc_relay_smoke.c ds4_v100_context.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_hc_relay_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_gpu.h ds4_pack.h ds4_source_formats.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ifeq ($(UNAME_S),Darwin)
tests/cuda_bf16_probe:
	@echo "tests/cuda_bf16_probe requires a CUDA build"
	@exit 2
tests/cuda_v100_context_smoke:
	@echo "tests/cuda_v100_context_smoke requires a CUDA build"
	@exit 2
tests/cuda_source_dtypes_smoke:
	@echo "tests/cuda_source_dtypes_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_prefill_kv_smoke:
	@echo "tests/cuda_v100_prefill_kv_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_compressor_bridge_smoke:
	@echo "tests/cuda_v100_compressor_bridge_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_projection_attention_smoke:
	@echo "tests/cuda_v100_projection_attention_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_bounded_logits_smoke:
	@echo "tests/cuda_v100_bounded_logits_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_mxfp4_moe_smoke:
	@echo "tests/cuda_v100_mxfp4_moe_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_descriptor_bound_ffn_smoke:
	@echo "tests/cuda_v100_descriptor_bound_ffn_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_descriptor_bound_attention_smoke:
	@echo "tests/cuda_v100_descriptor_bound_attention_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_integrated_layer_smoke:
	@echo "tests/cuda_v100_integrated_layer_smoke requires a CUDA build"
	@exit 2
tests/cuda_v100_stage_scheduler_smoke:
	@echo "tests/cuda_v100_stage_scheduler_smoke requires a CUDA build"
	@exit 2
tests/cuda_hc_relay_smoke:
	@echo "tests/cuda_hc_relay_smoke requires a CUDA build"
	@exit 2
else
tests/cuda_bf16_probe: tests/cuda_bf16_probe.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_context_smoke: tests/cuda_v100_context_smoke.o ds4_v100_context.o ds4_v100_context_cuda.o ds4_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_source_dtypes_smoke: tests/cuda_source_dtypes_smoke.o ds4_cuda.o ds4_source_formats.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_prefill_kv_smoke: tests/cuda_v100_prefill_kv_smoke.o ds4_cuda.o ds4_source_formats.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_compressor_bridge_smoke: tests/cuda_v100_compressor_bridge_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_projection_attention_smoke: tests/cuda_v100_projection_attention_smoke.o ds4_cuda.o ds4_source_formats.o ds4_v100_context.o ds4_v100_context_cuda.o ds4_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_bounded_logits_smoke: tests/cuda_v100_bounded_logits_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_mxfp4_moe_smoke: tests/cuda_v100_mxfp4_moe_smoke.o ds4_cuda.o ds4_source_formats.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_descriptor_bound_ffn_smoke: tests/cuda_v100_descriptor_bound_ffn_smoke.o ds4_cuda.o ds4_v100_layer_state.o ds4_source_formats.o ds4_v100_context.o ds4_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_descriptor_bound_attention_smoke: tests/cuda_v100_descriptor_bound_attention_smoke.o ds4_cuda.o ds4_v100_layer_state.o ds4_source_formats.o ds4_v100_context.o ds4_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_integrated_layer_smoke: tests/cuda_v100_integrated_layer_smoke.o ds4_cuda.o $(V100_LAYER_EXECUTE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_v100_stage_scheduler_smoke: tests/cuda_v100_stage_scheduler_smoke.o ds4_cuda.o $(V100_SCHEDULER_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
tests/cuda_hc_relay_smoke: tests/cuda_hc_relay_smoke.o ds4_v100_context.o ds4_v100_context_cuda.o ds4_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4_cpu ds4_native ds4_server_test ds4_test *.o tests/*.o tests/cuda_long_context_smoke tests/cuda_bf16_probe tests/cuda_v100_context_smoke tests/cuda_source_dtypes_smoke tests/cuda_v100_prefill_kv_smoke tests/cuda_v100_compressor_bridge_smoke tests/cuda_v100_projection_attention_smoke tests/cuda_v100_bounded_logits_smoke tests/cuda_v100_mxfp4_moe_smoke tests/cuda_v100_descriptor_bound_ffn_smoke tests/cuda_v100_descriptor_bound_attention_smoke tests/cuda_v100_integrated_layer_smoke tests/cuda_v100_stage_scheduler_smoke tests/cuda_hc_relay_smoke tests/pack_index_smoke tests/gpu_arena_smoke tests/bf16_probe_smoke tests/v100_context_smoke tests/v100_layer_binding_smoke tests/v100_layer_state_smoke tests/source_dtypes_smoke tools/*.o tools/ds4-v100-plan tools/ds4-v100-pack tools/ds4-v100-residency-smoke tools/ds4-v100-context-smoke tools/ds4-v100-layer-descriptor-gate tools/ds4-source-oracle-vector
