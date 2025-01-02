from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = "OpenGVLab/InternVL2-Llama3-76B"
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192, tp=2, cache_max_entry_count=0.4))

image = load_image("./human-pose.jpg")
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(("describe this image", image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat("What is the woman doing?", session=sess, gen_config=gen_config)
print(sess.response.text)
