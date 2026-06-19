# This is only meant for my personal use on AMD cluster. Refer to setup.md for a machine agnostic setup.
# Start SkyRL tinker server on AMD GPUs

## Pull apptainer

```bash
export APPTAINER_CACHEDIR="/work1/grahamneubig/adityabs/apptainer_cache_dir"
mkdir $APPTAINER_CACHEDIR 
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir $APPTAINER_TMPDIR
apptainer pull /work1/grahamneubig/adityabs/vllm_vllm-openai-rocm_v0.20.2.sif docker://docker.io/vllm/vllm-openai-rocm:v0.20.2
```

## Install Dependencies
```bash
cd /work1/grahamneubig/adityabs/skyrl_tinker_amd/
apptainer shell \
    --hostname "$(hostname -s)" \
    --bind "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem,/etc/pki/tls/certs:/etc/pki/tls/certs" \
    /work1/grahamneubig/adityabs/vllm_vllm-openai-rocm_v0.20.2.sif 
cp pyproject.toml pyproject.toml.bak
cp pyproject.other.toml pyproject.toml
python -m venv --system-site-packages /work1/grahamneubig/adityabs/.tinker_venv/
source /work1/grahamneubig/adityabs/.tinker_venv/bin/activate
/work1/grahamneubig/adityabs/.tinker_venv/bin/python -m pip install -e '.[fsdp,tinker]'
/work1/grahamneubig/adityabs/.tinker_venv/bin/python -m pip install -U ray[all]==2.51.1
/work1/grahamneubig/adityabs/.tinker_venv/bin/python -m pip install flash-linear-attention[rocm]
/work1/grahamneubig/adityabs/.tinker_venv/bin/python -m pip install orjson torchdata
cp pyproject.toml.bak pyproject.toml
rm -rf pyproject.toml.bak
```

## Launch Fresh Apptainer Shell

```bash
cd /work1/grahamneubig/adityabs/skyrl_tinker_amd/
apptainer shell \
    --hostname "$(hostname -s)" \
    --bind "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem,/etc/pki/tls/certs:/etc/pki/tls/certs" \
    /work1/grahamneubig/adityabs/vllm_vllm-openai-rocm_v0.20.2.sif 
source /work1/grahamneubig/adityabs/.tinker_venv/bin/activate
rm -rf skyrl/tinker/tinker.db*
rm -rf /tmp/skyrl*
```

## Launch the Tinker Server
```bash
HOME=/tmp SKYRL_RAY_NUM_CPUS=64 uv run --active --no-sync --extra tinker --extra fsdp \
    -m skyrl.tinker.api \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --backend fsdp \
    --port 9000 \
    --backend-config '{
        "trainer.placement.colocate_all": false,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.policy_num_gpus_per_node": 2,
        "trainer.micro_train_batch_size_per_gpu": 32,
        "trainer.micro_forward_batch_size_per_gpu": 32,

        "generator.inference_engine.num_engines": 6,
        "generator.inference_engine.max_num_batched_tokens": 32768,
        "generator.inference_engine.enable_ray_prometheus_stats": false,
        "generator.inference_engine.gpu_memory_utilization": 0.8,
        "generator.inference_engine.max_num_seqs": 256,
        "generator.inference_engine.engine_init_kwargs.max_model_len": 32768
    }'
```

## Forward Port to Babel
-NOTE: Edit /home1/adityabs/.ssh/config and add the right babel-compute-node name if needed.
```bash
ssh -N -R 0.0.0.0:9000:localhost:9000 babel-compute-node
```