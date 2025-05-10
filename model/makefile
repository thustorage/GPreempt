
SUBDIRS = ${shell ls -d */ | sed 's/\///' -}
CURRENT_PATH = ${shell pwd}
CURRENT_DIR = ${shell basename $(CURRENT_PATH)}

USE_CUDA = $(shell command -v nvcc >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(USE_CUDA), yes)
	COMPILER = nvcc
	FLAGS = -arch=sm_80 -O3 --cubin
else
	COMPILER = hipcc
	FLAGS = --offload-arch=gfx908 --genco -O3
endif

all: 
	@for dir in $(SUBDIRS); do \
		make -C $$dir -f ../makefile build_subdir;\
	done

build_subdir: mod.cubin mod.tr.cu mod.tr.cubin mod.be.cu mod.be.cubin

%.cubin: %.cu
	$(COMPILER) $(FLAGS) $< -o $@

%.tr.cu: %.cu
	python ../../scripts/compile/convert_to_blp.py $< $@

%.be.cu: %.cu
	python ../../scripts/compile/kernel_transform_be.py $< $@

%.be.cubin: %.be.cu
	$(COMPILER) $(FLAGS) $< -o $@

clean: 
	@for dir in $(SUBDIRS); do \
		make -C $$dir -f ../makefile clean_subdir;\
	done

clean_subdir:
	rm -f *.cubin *.tr.cu *.tr.cubin *.be.cu *.be.cubin