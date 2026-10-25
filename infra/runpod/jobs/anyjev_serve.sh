# Source with MODEL, KIND (tacit|decider), MEM (vLLM gpu fraction), optional VPORT/APORT and AJ_GPU
# (serve on that GPU only, e.g. a second GPU when training needs all of GPU 0).
# Starts `vllm serve` (anyjev venv) and monitors.anyjev_server in the background, waits until ready.
: "${MODEL:?}" "${KIND:?}" "${MEM:=0.3}" "${VPORT:=8000}" "${APORT:=8100}" "${MAXLEN:=8192}"
AJ=$AJ_VENV
export PATH=$AJ/bin:$PATH   # vLLM 0.29 JIT warmups shell out to `ninja`
[ -x $AJ/bin/ninja ] || uv pip install --python $AJ/bin/python -q ninja
mkdir -p $WS/jobs/anyjev
CUDA_VISIBLE_DEVICES=${AJ_GPU:-${CUDA_VISIBLE_DEVICES:-0}} $AJ/bin/vllm serve "$MODEL" --host 127.0.0.1 --port $VPORT --gpu-memory-utilization $MEM \
    --max-model-len $MAXLEN --dtype bfloat16 --enable-prefix-caching --max-logprobs 64 \
    > $WS/jobs/anyjev/vllm_$(basename $MODEL).log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do curl -sf 127.0.0.1:$VPORT/v1/models >/dev/null && break; sleep 5; done
curl -sf 127.0.0.1:$VPORT/v1/models >/dev/null || { echo "vllm serve failed"; tail -30 $WS/jobs/anyjev/vllm_$(basename $MODEL).log; exit 1; }
PYTHONPATH=$CODE $AJ/bin/python -m monitors.anyjev_server --kind $KIND --model "$MODEL" \
    --upstream http://127.0.0.1:$VPORT --port $APORT > $WS/jobs/anyjev/server_$(basename $MODEL).log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 60); do curl -sf 127.0.0.1:$APORT/ >/dev/null && break; sleep 2; done
curl -sf 127.0.0.1:$APORT/ || { echo "anyjev server failed"; tail -30 $WS/jobs/anyjev/server_$(basename $MODEL).log; exit 1; }
echo
stamp "AnyJev $KIND $MODEL ready (vllm pid $VLLM_PID, server pid $SERVER_PID)"
