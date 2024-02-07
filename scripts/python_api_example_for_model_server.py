import time
from neural_speed import Model as CppModel
import neural_speed.gptj_cpp as cpp
from transformers import AutoTokenizer

prompts = [
    "she opened the door and see",
    "tell me 10 things about jazz music",
    "What is the meaning of life?",
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"
    " The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles."
    "And by opposing end them. To die—to sleep,",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Intel?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano.",
][:4]

model_name = "/home/wangzhe/dingyi/models/finetuned-gptj"  # model_name from huggingface or local model path
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

res_collect = [None for _ in prompts]


def f_response(res, working):
    ret_token_ids = [r.token_ids for r in res]
    for r in res:
        res_collect[r.id] = r.token_ids
    ans = tokenizer.batch_decode(ret_token_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    print(f"working_size: {working}, ans:", flush=True)
    for a in ans:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(a)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", flush=True)


# please set your corresponding local neural_speed low-bits model file
model_path = "/home/wangzhe/dingyi/models/finetuned-gptj-pr83-q4-j-int8-pc.bin"
# added_count = 0
# s = cpp.ModelServer(f_response,                      # reponse function (deliver generation results and current reamin working size in server)
#                     model_path,                      # model_path
#                     ctx_size=2048,
#                     max_new_tokens=128,              # global query max generation token length
#                     num_beams=4,                     # global beam search related generation parameters
#                     min_new_tokens=30,               # global beam search related generation parameters (default: 0)
#                     early_stopping=True,             # global beam search related generation parameters (default: False)
#                     continuous_batching=True,        # turn on continuous batching mechanism (default: True)
#                     # also return prompt token ids in generation results (default: False)
#                     return_prompt=False,
#                     # number of threads in model evaluate process (please bind cores if need)
#                     threads=56,
#                     max_request_num=8,               # maximum number of running requests (or queries, default: 8)
#                     print_log=False,                  # print server running logs (default: False)
#                     model_scratch_enlarge_scale=1,  # model memory scratch enlarge scale (default: 1)
#                     )
# for i in (range(len(prompts))):
#     p_token_ids = tokenizer(prompts[i], return_tensors='pt').input_ids.tolist()
#     s.issueQuery([cpp.Query(i, p_token_ids)])
#     added_count += 1
#     time.sleep(0.5)  # adjust query sending time interval

# # recommend to use time.sleep in while loop to exit program
# # let cpp server owns more resources
# while (added_count != len(prompts) or not s.Empty()):
#     time.sleep(1)
# del s
# print("should finished\n\n\n")
# result_str = tokenizer.batch_decode(res_collect, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# for i, s in enumerate(result_str):
#     print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     print(prompts[i])
#     print(s.replace('\n', r'\n')[:100], "...", flush=True)


cpp_model = CppModel()
cpp_model.model_type = 'gptj'
cpp_model.init_from_bin(
    cpp_model.model_type,
    model_path,
    max_new_tokens=128,
    n_batch=2048,
    ctx_size=2048,
    threads=56,
    num_beams=4,
    do_sample=False,
    min_new_tokens=30,
    early_stopping=True,
    n_keep=0,
    n_discard=-1,
    shift_roped_k=False,
    batch_size=4,
    pad_token=-1,
    memory_dtype="auto",
    continuous_batching=True,
    max_request_num=4,
    model_scratch_enlarge_scale=max(1, 4 / 6.)  # double when batchsize=12 for safety
)
res_collect = cpp_model.model.generate([tokenizer.encode(p) for p in prompts[:4]])
result_str = tokenizer.batch_decode(res_collect, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("\n\n\nReference:")
for i, s in enumerate(result_str):
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(prompts[i])
    print(s.replace('\n', r'\n')[:100], "...", flush=True)