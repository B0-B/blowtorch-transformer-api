from blowtorch import client
import subprocess

# try:
#     line_as_bytes = subprocess.check_output("rocm-smi --showproductname", shell=True)
# except:
#     line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
# return line_as_bytes.decode("utf-8")
client(hugging_face_path='TheBloke/Llama-2-7B-Chat-GGML', device='cpu', model_type="llama").bench()