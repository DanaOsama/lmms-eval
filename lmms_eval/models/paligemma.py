import os
import uuid
import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

"""
Command to run:
----------------
python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model paligemma \
    --model_args model_id_name=google/paligemma-3b-ft-ocrvqa-896,prefix="answer en"\
    --tasks textvqa,stvqa,infovqa,docvqa\
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix paligemma_ocrvqa_896 \
    --output_path ./logs/

-----
Different checkpoints to be used:
    - google/paligemma-3b-ft-ocrvqa-896 
    - google/paligemma-3b-ft-stvqa-896
    - google/paligemma-3b-ft-textvqa-896

Note: Login to huggingface and make sure to acknowledge 
    the terms and conditions on HF's model card.
"""


@register_model("paligemma")
class PaliGemma(lmms):
    """
    PaliGemma Model
    https://github.com/google-research/big_vision/tree/main/big_vision/evaluators/proj/paligemma
    """

    def __init__(
        self,
        model_id_name: str = "google/paligemma-3b-ft-ocrvqa-896",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        prefix="answer en",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
        
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(model_id_name).eval()
        self._processor = AutoProcessor.from_pretrained(model_id_name)
        self._processor.tokenizer.padding_side = "left"
        self._tokenizer = self._processor.tokenizer
        # self._model = AutoModelForCausalLM.from_pretrained(model_id_name, device=self._device, trust_remote_code=trust_remote_code, torch_dtype=dtype)
        # self._tokenizer = AutoTokenizer.from_pretrained(model_id_name, trust_remote_code=trust_remote_code)
        # self._tokenizer.padding_side = "left"
        # self._tokenizer.pad_token_id = self._tokenizer.eos_token 

        # prefix for paligemma is "answer en" for VQA tasks in English, or "caption en\n" for captioning.
        self.prefix = prefix
        self._config = self._model.config
        # self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size_per_gpu > 1 is not supported for now."

        self.use_cache = use_cache
        # if accelerator.num_processes > 1:
        #     assert accelerator.distributed_type in [
        #         DistributedType.FSDP,
        #         DistributedType.MULTI_GPU,
        #         DistributedType.DEEPSPEED
        #     ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
        #     if accelerator.distributed_type == DistributedType.FSDP:
        #         self._model = accelerator.prepare(self.model)
        #     else:
        #         self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
        #     self.accelerator = accelerator
        #     if self.accelerator.is_local_main_process:
        #         eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        #     self._rank = self.accelerator.local_process_index
        #     self._world_size = self.accelerator.num_processes
        # else:
        #     self.model.to(self._device)
        #     self._rank = 0
        #     self._world_size = 1
        if accelerator.num_processes > 1 and device == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model
    
    @property
    def processor(self):
        return self._processor

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for Paligemma.")
    
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

# TODO: Check if we need those functions
#     def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
#         """ """
#         add_special_tokens = False if add_special_tokens is None else add_special_tokens
#         encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
#         # left-truncate the encoded context to be at most `left_truncate_len` tokens long
#         if left_truncate_len:
#             encoding = encoding[-left_truncate_len:]
#     return encoding

    # def tok_decode(self, tokens):
    #     return self.tokenizer.decode(tokens)

    # def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
    #     res = []
    #     pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

    #     for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
    #         # encode, pad, and truncate contexts for this batch
    #         if type(doc_to_target) == str:
    #             continuation = doc_to_target
    #         else:
    #             continuation = doc_to_target(self.task_dict[task][split][doc_id])
    #         visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
    #         visuals = self.flatten(visuals)
    #         query = []
    #         visual_paths = []
    #         for visual in visuals:
    #             name = uuid.uuid4().hex.upper()[0:6]
    #             visual.save(f"/tmp/{name}.png")
    #             visual_paths.append(f"/tmp/{name}.png")
    #             query.append({"image": f"/tmp/{name}.png"})

    #         # Make a copy for query to save context (text that needs to be masked)
    #         context_query = [_ for _ in query]
    #         context_query.append({"text": contexts})
    #         query.append({"text": contexts + continuation})

    #         context_query = self._tokenizer.from_list_format(context_query)
    #         query = self._tokenizer.from_list_format(query)

    #         raw_contxt_text, context_tokens = make_context(
    #             self._tokenizer, context_query, history=None, system="You are a helpful assistant", max_window_size=self.model.generation_config.max_window_size, chat_format=self.model.generation_config.chat_format
    #         )
    #         context_tokens = torch.tensor([context_tokens])

    #         raw_continuation_text, continuation_tokens = make_context(
    #             self._tokenizer, query, history=None, system="You are a helpful assistant", max_window_size=self.model.generation_config.max_window_size, chat_format=self.model.generation_config.chat_format
    #         )
    #         continuation_tokens = torch.tensor([continuation_tokens]).to(self.model.device)
    #         attn_mask = torch.ones_like(continuation_tokens).to(self.model.device)
    #         labels = continuation_tokens.clone().to(self.model.device)
    #         labels[:, : context_tokens.shape[1]] = -100
    #         with torch.inference_mode():
    #             outputs = self.model(input_ids=continuation_tokens, labels=labels, attention_mask=attn_mask)
    #         loss = outputs.loss
    #         logits = outputs["logits"]
    #         greedy_tokens = logits.argmax(dim=-1)
    #         cont_toks = continuation_tokens[:, context_tokens.shape[1] :]
    #         greedy_tokens = greedy_tokens[:, context_tokens.shape[1] : continuation_tokens.shape[1]]  # [1, seq]
    #         max_equal = (greedy_tokens == cont_toks).all()
    #         res.append((float(loss.item()), bool(max_equal)))
    #         pbar.update(1)

    #     pbar.close()
    #     return res

    # def flatten(self, input):
    #     new_list = []
    #     for i in input:
    #         for j in i:
    #             new_list.append(j)
    #     return new_list

    # 🔹 What is a "Request" in `lmms_eval`?
# A request is an instruction for the model to perform a task.
# Each request is represented as an `Instance` object, which contains:
# - The input (e.g., a text prompt or question).
# - Metadata (e.g., dataset details, document ID).
# - Generation settings (e.g., temperature, max tokens).
#
# Requests tell the model:
# - What to do (e.g., generate text, calculate log-likelihood).
# - How to do it (e.g., using certain hyperparameters).

# 🔹 What are the Three Request Types?
# lmms_eval supports three types of requests:
# 1. `loglikelihood` - Computes the probability of a given continuation.
# 2. `multiple_choice` - Evaluates the model’s ability to pick the most likely answer among choices.
# 3. `generate_until` - Generates text until a stopping condition is met.

# 🔹 What is an `Instance`?
# A request is stored inside an `Instance` object, which contains:
# - `request_type`: One of `loglikelihood`, `multiple_choice`, or `generate_until`.
# - `args`: The actual input arguments for the model.
# - Other metadata like `idx` (index in batch) and extra settings.

# 🔹 Example `Instance` for `generate_until` request:
# Instance(
#     request_type="generate_until",
#     arguments=("What is in the image?", {"max_new_tokens": 128, "temperature": 0.7}, doc_to_visual, 101, "VQA", "test"),
#     idx=0
# )
# This tells the model:
# - Task: Generate text ("generate_until").
# - Input text: "What is in the image?"
# - Generation settings: { "max_new_tokens": 128, "temperature": 0.7 }
# - Image processing function: `doc_to_visual`
# - Dataset details: `doc_id = 101`, `task = "VQA"`, `split = "test"`

# 🔹 What are the `arguments` in a Request?
# For "generate_until", the `arguments` contain:
# 1. `contexts` - The text prompt for the model (could contain `<image>` tokens).
# 2. `all_gen_kwargs` - A dictionary of generation parameters (e.g., max tokens, temperature).
# 3. `doc_to_visual` - A function that retrieves and processes images related to the request.
# 4. `doc_id` - The document ID from the dataset.
# 5. `task` - The evaluation task (e.g., "VQA" for visual question answering).
# 6. `split` - The dataset split (e.g., "train", "test", "val").

# 🔹 Where is This Used?
# The function `construct_requests` creates `Instance` objects for different tasks.
# Then, in `generate_until`, we extract the arguments from the requests like this:
# contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
# This unpacks all the `arguments` from multiple `Instance` objects in a batch.

# 🔹 Summary:
# ✔ A request is an instruction to the model (e.g., "generate text" or "evaluate likelihood").
# ✔ Requests are stored as `Instance` objects with `request_type` and `arguments`.
# ✔ `generate_until` requests include:
#    - A text input (`contexts`).
#    - Generation parameters (`all_gen_kwargs`).
#    - Image processing function (`doc_to_visual`).
#    - Dataset details (`doc_id`, `task`, `split`).
# ✔ The function `construct_requests` creates these requests, and `generate_until` processes them. 🚀

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self._processor.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            # The variable contexts contains the text prompts.
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            # The task and split values are the same across all requests in a batch.
            # We extract just the first value since all should be identical.
            task = task[0]
            split = split[0]
            
            # TODO: Check how to deal with multiple images in one request
            # Convert document IDs into processed images using the first function in doc_to_visual
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # visual_paths = []

            # save images to /tmp, name generated by hash function
            # qwen accept image path. Have to do it here....
            # for visual in visuals:
            #     name = uuid.uuid4().hex.upper()[0:6]
            #     visual.save(f"/tmp/{name}.png")
            #     visual_paths.append(f"/tmp/{name}.png")

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            # breakpoint()
            until = [self.tokenizer.decode(self.eos_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]
            # text = " ".join([self.prefix, context])
            text = " ".join(['<image>' * len(visuals), self.prefix, context])


            # if isinstance(contexts, tuple):
            #     contexts = list(contexts)

            # # Some text prompts may include the placeholder "<image>" where an image was referenced.
            # # This removes "<image>" because the image is processed separately.
            # for i in range(len(contexts)):
            #     if "<image>" in contexts[i]:
            #         contexts[i] = contexts[i].replace("<image>", "")

            # Similar to llava, if visual paths has len 0
            # Then nothing will be executed
            # query = []
            # if len(visual_paths) == 0:
            #     for context in contexts:
            #         query.append({"text": context})
            # else:
            #     for visual_path, context in zip(visual_paths, contexts):
            #         query.append({"image": visual_path})
            #         query.append({"text": context})
            
            # TODO: Check the format of visuals acceptable for the Paligemma processor. I believe it is a PIL Image.
            # TODO: Check how to add support for multiple images.
            # questions = self.tokenizer.from_list_format(query)
            # input_ids = self.tokenizer(questions, return_tensors="pt", padding="longest")
            inputs = self.processor(images=visuals, text=text, return_tensors="pt").to(self.device, self.model.dtype)

            # preconfigure gen_kwargs with defaults
            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size] # set size of the first image
                except:
                    gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 20 # Default max output tokens
                # gen_kwargs["max_new_tokens"] = 6144 # Default max output tokens
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 1 # Greedy decoding by default is temp=0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None # No nucleus sampling
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # pad_token = self.tokenizer.pad_token if self.tokenizer.pad_token_id is not None else self.tokenizer.eod_id
            # try:
            #     output = self.model.generate(
            #         **inputs, # Unpack the dictionary 'inputs' as keyword arguments (likely contains 'input_ids' and 'attention_mask')
            #         eos_token_id=self.eos_token_id,
            #         pad_token_id=self.eos_token_id,
            #         do_sample=True if gen_kwargs["temperature"] > 0 else False,
            #         temperature=gen_kwargs["temperature"],
            #         top_p=gen_kwargs["top_p"],
            #         num_beams=gen_kwargs["num_beams"],
            #         max_new_tokens=gen_kwargs["max_new_tokens"],
            #         use_cache=self.use_cache,
            #         # kwargs=gen_kwargs
            #     )

            # except Exception as e:
            #     eval_logger.error(f"Error {e} in generating")
            #     output = ""

            output = self.model.generate(
                    **inputs, # Unpack the dictionary 'inputs' as keyword arguments (likely contains 'input_ids' and 'attention_mask')
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.eos_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    # kwargs=gen_kwargs
            )

            # This line does what the following 3 lines of code do, but I kept the below so I understand it better
            # breakpoint()
            # text_outputs = self.processor.decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ]
            # text_outputs = self.processor.decode(output[0], skip_special_tokens=True).replace(text, "")
            decoded_text = self.processor.decode(output[0], skip_special_tokens=True)
            prompt_text = self.processor.decode(inputs.input_ids[0], skip_special_tokens=True)

            # Remove the prompt from the generated output
            text_outputs = decoded_text[len(prompt_text):].strip()
            
            # print("Text_output:", text_outputs)
            # Decode the first generated sequence, removing special tokens
            # decoded_text = self.processor.decode(output[0], skip_special_tokens=True)
            # breakpoint()
            # # Get the number of tokens in the original input prompt
            # prompt_length = inputs.input_ids.shape[1]

            # # Slice the decoded text to keep only the newly generated part (excluding the prompt)
            # text_outputs = decoded_text[prompt_length:]

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            
            # Update the progress bar by 1 step to reflect progress in processing requests.
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
