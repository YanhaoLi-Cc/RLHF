set -x

pip install -r requirements.txt

DS_BUILD_CPU_ADAM=1 pip install deepspeed==0.16.2