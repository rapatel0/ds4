#include "engine/layer_execute.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "engine/layer_execute_core.inc"
#include "engine/layer_execute_attention.inc"
#include "engine/layer_execute_ffn.inc"
