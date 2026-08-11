"""Implement HuggingfaceModel models."""
import copy
import logging
import os
from collections import Counter

import accelerate
import torch
from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES

HF_CACHE_DIR = '/vol/bitbucket/bhw22/miniconda/envs/uncertainty_components/lib/python3.11/site-packages/cache'


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer!
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModelAllEmbeddings(BaseModel):
    """HuggingfaceModel variant that returns all token embeddings when requested."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        print(model_name)
        if 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            if 'Llama-2' in model_name or 'Llama-3' in model_name:
                base = 'meta-llama'
                model_name = model_name + '-hf' if 'Llama-2' in model_name else model_name
            else:
                base = 'huggyllama'

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto",
                token_type_ids=None, cache_dir=HF_CACHE_DIR)

            llama65b = '65b' in model_name.lower() and base == 'huggyllama'
            llama2or3_70b = '70b' in model_name.lower() and base == 'meta-llama'

            if ('7b' in model_name or '13b' in model_name) or eightbit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto",
                    max_memory={0: '80GIB'}, cache_dir=HF_CACHE_DIR, **kwargs,)

            elif llama2or3_70b or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{model_name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json'],
                    cache_dir=HF_CACHE_DIR,
                )
                config = AutoConfig.from_pretrained(f"{base}/{model_name}", cache_dir=HF_CACHE_DIR)
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()
                if 'chat' in model_name:
                    max_mem = 17.5 * 4686198491
                else:
                    max_mem = 15 * 4686198491
                
                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16', skip_keys='past_key_values')

            else:
                raise ValueError

        elif 'mistral' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            if model_name.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,)}
                model_name = model_name[:-len('-8bit')]
            else:
                kwargs = {}

            model_id = f'mistralai/{model_name}'
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, device_map='auto', token_type_ids=None,
                    clean_up_tokenization_spaces=False, cache_dir=HF_CACHE_DIR)
            except Exception as e:
                logging.warning(f'Failed to load fast tokenizer for {model_name}. Trying slow tokenizer.')
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, device_map='auto', use_fast=False, token_type_ids=None,
                    clean_up_tokenization_spaces=False, cache_dir=HF_CACHE_DIR)

            if kwargs:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map='auto',
                    max_memory={0: '80GIB'},
                    cache_dir=HF_CACHE_DIR,
                    **kwargs,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map='auto',
                    max_memory={0: '80GIB'},
                    cache_dir=HF_CACHE_DIR,
                    torch_dtype=torch.float32,
                )

        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False, cache_dir=HF_CACHE_DIR)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                cache_dir=HF_CACHE_DIR,
                **kwargs,
            )
        elif 'phi' in model_name.lower():
            model_id = f'microsoft/{model_name}'  # e.g. Phi-3-mini-128k-instruct
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False, cache_dir=HF_CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                cache_dir=HF_CACHE_DIR,
            )
        elif 'gemma' in model_name:
            model_id = f'google/{model_name}'  # e.g. gemma-7b-it
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False, cache_dir=HF_CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                cache_dir=HF_CACHE_DIR,
            )
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = 4096 if 'Llama-2' in model_name else 2048

    
    def predict(
            self,
            input_data,
            temperature,
            return_full=False,
            return_latent=False,
            collect_attn_block_embeddings=False,
            collect_mlp_block_embeddings=False,
            collect_qkvo_embeddings=False,
            collect_concat_embeddings=False):

        if isinstance(input_data, tuple):
            logging.WARNING("INPUT IS A TUPLE.")
            input_data = input_data[0]

        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            if 'token_type_ids' in inputs:  # HF models seems has changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.info('temperature: %f', temperature)
        use_greedy = temperature == 0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_hidden_states": True,
            "do_sample": not use_greedy,
            "num_beams": 1,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": pad_token_id,
        }
        if not use_greedy:
            generation_kwargs["temperature"] = temperature
        logged_generation_kwargs = dict(generation_kwargs)
        logged_generation_kwargs["stopping_criteria"] = (
            "set" if stopping_criteria is not None else None
        )
        logging.info("Generation settings: %s", logged_generation_kwargs)

        attn_layer_outputs = []
        mlp_layer_outputs = []
        q_layer_outputs = []
        k_layer_outputs = []
        v_layer_outputs = []
        o_layer_outputs = []
        concat_layer_inputs = []
        hook_handles = []

        collect_attn = bool(return_latent and collect_attn_block_embeddings)
        collect_mlp = bool(return_latent and collect_mlp_block_embeddings)
        collect_qkvo = bool(return_latent and collect_qkvo_embeddings)
        collect_concat = bool(return_latent and collect_concat_embeddings)
        if collect_attn or collect_mlp or collect_qkvo or collect_concat:
            layers = getattr(getattr(self.model, "model", None), "layers", None)
            if layers is None:
                logging.warning(
                    "Model does not expose model.layers; skipping attn/mlp/qkvo/concat embedding collection."
                )
                collect_attn = False
                collect_mlp = False
                collect_qkvo = False
                collect_concat = False
            else:
                if collect_attn:
                    attn_layer_outputs = [[] for _ in range(len(layers))]
                if collect_mlp:
                    mlp_layer_outputs = [[] for _ in range(len(layers))]
                if collect_qkvo:
                    q_layer_outputs = [[] for _ in range(len(layers))]
                    k_layer_outputs = [[] for _ in range(len(layers))]
                    v_layer_outputs = [[] for _ in range(len(layers))]
                    o_layer_outputs = [[] for _ in range(len(layers))]
                if collect_concat:
                    concat_layer_inputs = [[] for _ in range(len(layers))]

                def _extract_hidden_tensor(module_output):
                    if isinstance(module_output, (tuple, list)):
                        if len(module_output) == 0:
                            return None
                        return module_output[0]
                    return module_output

                def _make_attn_hook(layer_idx):
                    def _hook(_, __, module_output):
                        tensor = _extract_hidden_tensor(module_output)
                        if tensor is None:
                            return
                        attn_layer_outputs[layer_idx].append(tensor.detach())
                    return _hook

                def _make_mlp_hook(layer_idx):
                    def _hook(_, __, module_output):
                        tensor = _extract_hidden_tensor(module_output)
                        if tensor is None:
                            return
                        mlp_layer_outputs[layer_idx].append(tensor.detach())
                    return _hook

                def _make_qkvo_hook(layer_outputs, layer_idx):
                    def _hook(_, __, module_output):
                        tensor = _extract_hidden_tensor(module_output)
                        if tensor is None:
                            return
                        layer_outputs[layer_idx].append(tensor.detach())
                    return _hook

                def _extract_pre_hook_input(module_input):
                    if not isinstance(module_input, (tuple, list)) or len(module_input) == 0:
                        return None
                    tensor = module_input[0]
                    if isinstance(tensor, (tuple, list)):
                        if len(tensor) == 0:
                            return None
                        tensor = tensor[0]
                    return tensor

                def _make_concat_pre_hook(layer_idx):
                    def _hook(_, module_input):
                        tensor = _extract_pre_hook_input(module_input)
                        if tensor is None:
                            return
                        concat_layer_inputs[layer_idx].append(tensor.detach())
                    return _hook

                # Gemma3: cache post-norm residual writes when present; else self_attn/mlp.
                # Gate on post_feedforward_layernorm so Mistral's pre-MLP
                # post_attention_layernorm is not mistaken for a post-attn write norm.
                use_gemma_post_norm_hooks = any(
                    hasattr(layer, "post_feedforward_layernorm") for layer in layers
                )
                if collect_attn or collect_mlp:
                    logging.info(
                        "Attn/mlp cache hook targets: %s",
                        (
                            "post_attention_layernorm / post_feedforward_layernorm"
                            if use_gemma_post_norm_hooks
                            else "self_attn / mlp"
                        ),
                    )

                for layer_idx, layer in enumerate(layers):
                    if collect_attn:
                        if use_gemma_post_norm_hooks and hasattr(
                            layer, "post_attention_layernorm"
                        ):
                            attn_mod = layer.post_attention_layernorm
                        else:
                            attn_mod = getattr(layer, "self_attn", None)
                        if attn_mod is not None:
                            hook_handles.append(
                                attn_mod.register_forward_hook(_make_attn_hook(layer_idx))
                            )
                    if collect_mlp:
                        if use_gemma_post_norm_hooks and hasattr(
                            layer, "post_feedforward_layernorm"
                        ):
                            mlp_mod = layer.post_feedforward_layernorm
                        else:
                            mlp_mod = getattr(layer, "mlp", None)
                        if mlp_mod is not None:
                            hook_handles.append(
                                mlp_mod.register_forward_hook(_make_mlp_hook(layer_idx))
                            )
                    if collect_qkvo and hasattr(layer, "self_attn"):
                        self_attn = layer.self_attn
                        if hasattr(self_attn, "q_proj"):
                            hook_handles.append(
                                self_attn.q_proj.register_forward_hook(_make_qkvo_hook(q_layer_outputs, layer_idx))
                            )
                        if hasattr(self_attn, "k_proj"):
                            hook_handles.append(
                                self_attn.k_proj.register_forward_hook(_make_qkvo_hook(k_layer_outputs, layer_idx))
                            )
                        if hasattr(self_attn, "v_proj"):
                            hook_handles.append(
                                self_attn.v_proj.register_forward_hook(_make_qkvo_hook(v_layer_outputs, layer_idx))
                            )
                        if hasattr(self_attn, "o_proj"):
                            hook_handles.append(
                                self_attn.o_proj.register_forward_hook(_make_qkvo_hook(o_layer_outputs, layer_idx))
                            )
                    if collect_concat and hasattr(layer, "self_attn"):
                        self_attn = layer.self_attn
                        if hasattr(self_attn, "o_proj"):
                            hook_handles.append(
                                self_attn.o_proj.register_forward_pre_hook(_make_concat_pre_hook(layer_idx))
                            )

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
        finally:
            for handle in hook_handles:
                handle.remove()

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        # Remove input tokens from decoded_tokens
        decoded_tokens = [self.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in outputs.sequences[0]]
        sliced_decoded_tokens = decoded_tokens[n_input_token:]

        logging.info(f'Length of generated output (with special tokens) (sliced_decoded_tokens): {len(sliced_decoded_tokens)}')
        logging.info(f'Full length (input prompt + generated output) (with special tokens) (decoded_tokens): {len(decoded_tokens)}')

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # if access idx is larger/equal
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        if return_latent:
            # Collect all embeddings (per-layer tensors) and the corresponding inputs.
            all_embeddings = []
            for h in hidden:
                # h may be a tuple/list of per-layer tensors or a tensor
                if isinstance(h, (list, tuple)):
                    stacked = torch.stack([layer for layer in h])
                else:
                    stacked = h
                all_embeddings.append(stacked.cpu())

            # Stack second last token embeddings from all layers 
            if len(hidden) == 1:  # FIX: runtime error for mistral-7b on bioasq
                sec_last_input = hidden[0]
            elif ((n_generated - 2) >= len(hidden)):
                sec_last_input = hidden[-2]
            else:
                # TODO: it used to be a mistake
                sec_last_input = hidden[n_generated - 1]
            emb_sec_last_token = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
    
            # Get the last input token embeddings (before generated tokens)
            last_tok_bef_gen_input = hidden[0]
            emb_tok_bef_gen = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

            all_attn_embeddings = None
            all_mlp_embeddings = None
            all_q_embeddings = None
            all_k_embeddings = None
            all_v_embeddings = None
            all_o_embeddings = None
            all_concat_embeddings = None

            def _pack_layerwise_outputs(layer_outputs, name):
                if not layer_outputs:
                    return None
                if any(len(per_layer) == 0 for per_layer in layer_outputs):
                    logging.warning(
                        "No %s outputs captured for at least one layer; skipping %s collection.",
                        name, name
                    )
                    return None

                min_calls = min(len(per_layer) for per_layer in layer_outputs)
                expected_calls = len(hidden)
                if min_calls < expected_calls:
                    logging.warning(
                        "Captured %d %s calls but hidden has %d steps. Using %d captured steps.",
                        min_calls, name, expected_calls, min_calls
                    )
                    use_calls = min_calls
                    start_idx = 0
                else:
                    use_calls = expected_calls
                    start_idx = min_calls - expected_calls

                packed = []
                for call_idx in range(start_idx, start_idx + use_calls):
                    packed.append(torch.stack([
                        per_layer[call_idx].cpu() for per_layer in layer_outputs
                    ]))
                return packed

            if collect_attn:
                all_attn_embeddings = _pack_layerwise_outputs(attn_layer_outputs, "attention")
            if collect_mlp:
                all_mlp_embeddings = _pack_layerwise_outputs(mlp_layer_outputs, "mlp")
            if collect_qkvo:
                all_q_embeddings = _pack_layerwise_outputs(q_layer_outputs, "q_proj")
                all_k_embeddings = _pack_layerwise_outputs(k_layer_outputs, "k_proj")
                all_v_embeddings = _pack_layerwise_outputs(v_layer_outputs, "v_proj")
                all_o_embeddings = _pack_layerwise_outputs(o_layer_outputs, "o_proj")
            if collect_concat:
                all_concat_embeddings = _pack_layerwise_outputs(concat_layer_inputs, "concat")

        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        hidden_states = ()

        if return_latent:
            hidden_states += (
                emb_sec_last_token,
                emb_tok_bef_gen,
                all_embeddings,
                all_attn_embeddings,
                all_mlp_embeddings,
                all_q_embeddings,
                all_k_embeddings,
                all_v_embeddings,
                all_o_embeddings,
                all_concat_embeddings,
            )
        else:
            hidden_states += (None, None, None, None, None, None, None, None, None, None)

        return_values = (sliced_answer, log_likelihoods, hidden_states, sliced_decoded_tokens)

        return return_values

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()

    def get_perplexity(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        tokenized_data = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        with torch.no_grad():
            model_output_true = self.model(tokenized_data, labels=tokenized_data)

        perplexity = - model_output_true.loss.item()


        return perplexity
