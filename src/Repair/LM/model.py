import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers import T5ForConditionalGeneration


# Adopted from https://github.com/huggingface/transformers/pull/14897
class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any([stop_string in decoded_generation for stop_string in self.eof_strings])
            if finished and index not in self.end_length:  # ensures first time we see it
                for stop_string in self.eof_strings:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(input_ids[index, # get length of actual generation
                                                     self.start_length:
                                                     -len(self.tokenizer.encode(stop_string, add_special_tokens=False,
                                                                                return_tensors='pt')[0])])
            done.append(finished)
        return all(done)


global_eof_stops = ['// Buggy Function', '// Fixed Function', '# Buggy Function', '# Fixed Function',
                    '/* Buggy Function */', '/* Fixed Function */', '<|endoftext|>']


class GPT2(object):
    def __init__(self, batch_size: int = 1, pretrained: str = 'gpt2', stop="", weight=None):
        print("Initializing a GPT-2 based model: {} ...".format(pretrained))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # GPT-NeoX issue: https://github.com/huggingface/transformers/issues/17452
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float16)
        if weight == 'float16':
            print("Switching to float16 ...")
            self.model = self.model.half()
        elif weight == 'bfloat16':  # neo 2.7b can be loaded using only 8 gb with bfloat16
            print("Switching to bfloat16 ...")
            self.model = self.model.to(torch.bfloat16)

        self.model = self.model.to(self.device)
        self.max_length = 1024  # default context size of 1024
        # use max position embeddings to determine max length
        if 'max_position_embeddings' in self.model.config.to_dict():
            self.max_length = self.model.config.to_dict()['max_position_embeddings']
        elif 'n_positions' in self.model.config.to_dict():
            self.max_length = self.model.config.to_dict()['n_positions']
        print("Max length: {}".format(self.max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.stop = stop
        # TODO: add batch size
        self.batch_size = batch_size

    def check_input(self, prompt: str, buggy_func: str):
        # Check if prompt + fix_function=approx(buggy_func) will be longer than the max length
        input_tokens = self.tokenizer.encode(prompt + "\n" + buggy_func, return_tensors='pt')
        if len(input_tokens[0]) > self.max_length:
            return False
        return True

    def model_predict(self, prompt: str, buggy_func: str, do_sample=False, num_samples=10000):
        if not self.check_input(prompt, buggy_func):
            return False, False, None, None  # If the input is too long, return False
        input_tokens = self.tokenizer.encode(prompt, return_tensors='pt').repeat(min(self.batch_size, num_samples), 1)
        input_tokens = input_tokens.to(self.device)
        sc = StoppingCriteriaList([EndOfFunctionCriteria(start_length=len(input_tokens[0]),
                                                         eof_strings=[self.stop] + global_eof_stops,
                                                         tokenizer=self.tokenizer)])

        with torch.no_grad():
            raw_o = self.model.generate(input_tokens,
                                        max_length=min(self.max_length, len(input_tokens[0]) +
                                                       int(2*len(self.tokenizer.encode(buggy_func, return_tensors='pt')[0]))),
                                        stopping_criteria=sc,
                                        do_sample=do_sample,
                                        top_p=0.95,
                                        temperature=0.8,
                                        output_scores=True,
                                        return_dict_in_generate=True,
                                        pad_token_id=self.tokenizer.eos_token_id)  # remove warning
            gen_sequences = raw_o.sequences[:, len(input_tokens[0]):]
            neg_logs = -torch.log(torch.stack(raw_o.scores, dim=1).softmax(-1))
            neg_logs = torch.gather(neg_logs, 2, gen_sequences[:, :, None]).squeeze(-1)
            t_outputs = self.tokenizer.batch_decode(gen_sequences, skip_special_tokens=False)
            outputs = []
            entropies = []
            for index, output in enumerate(t_outputs):
                min_index = 10000000
                for eof_string in [self.stop] + global_eof_stops:
                    if eof_string in output:
                        min_index = min(output.index(eof_string), min_index)
                        if index not in sc[0].end_length:
                            sc[0].end_length[index] = len(gen_sequences[index,
                                                          :-len(self.tokenizer.encode(eof_string,
                                                                                      add_special_tokens=False,
                                                                                      return_tensors='pt')[0])])

                if min_index != 10000000 and sc[0].end_length[index] != 0:
                    outputs.append(output[:min_index].strip())
                    entropies.append((neg_logs[index, :sc[0].end_length[index]].sum(-1).cpu().item() / sc[0].end_length[index],
                                      neg_logs[index, :sc[0].end_length[index]].sum(-1).cpu().item()))

        return True, len(outputs) > 0, outputs, entropies


global_infill_stops = ['# Provide a fix for the buggy function', '// Provide a fix for the buggy function']


class SpanLM(object):
    def __init__(self, pretrained: str = "", weight=None, batch_size=1):
        print("Initializing a SpanLM based model: {} ...".format(pretrained))
        self.pretrained = pretrained
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extra_end = None  # some models requires some ending tokens
        if 'Salesforce' in pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained)
            self.max_length = self.model.config.to_dict()['n_positions']
            self.infill_ph = "<extra_id_0>"
            self.END_ID = 2
            self.infill_ID = 32099
        elif 'facebook' in pretrained:
            if weight == 'float16':
                self.model = AutoModelForCausalLM.from_pretrained(pretrained, revision="float16", torch_dtype=torch.float16)
                self.model = self.model.half()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(pretrained)
            self.max_length = self.model.config.to_dict()['max_position_embeddings']
            self.infill_ph = "<|mask:0|>"
            self.extra_end = "<|mask:1|><|mask:0|>"
            # signals the end of a generated infill
            self.EOM = "<|endofmask|>"
            self.EOM_ID = 50517
            self.BOS = "<|endoftext|>"
            self.META_FILE = "<|/ file"
        else:
            raise NotImplementedError
        print("Max length: {}".format(self.max_length))
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.batch_size = batch_size

    def build_input(self, prefix: str, suffix: str):
        if self.extra_end:
            return prefix + self.infill_ph + suffix + self.extra_end
        return prefix + self.infill_ph + suffix

    def check_input(self, prefix: str, suffix: str, buggy: str = "", use_max_length: bool = False):
        if not use_max_length:
            input_tokens = self.tokenizer.encode(self.build_input(prefix, suffix), return_tensors='pt')
            if len(input_tokens[0]) + 50 > self.max_length:
                return False
        else:
            # Check if prompt + fix_function=approx(buggy_func) will be longer than the max length
            input_tokens = self.tokenizer.encode(prefix + buggy + suffix, return_tensors='pt')
            if len(input_tokens[0]) + 110 > self.max_length:
                return False
        return True

    def model_predict(self, prefix: str, suffix: str, do_sample=False, use_max_length=False, buggy: str = "", num_samples=1000):
        if not self.check_input(prefix, suffix, buggy, use_max_length):
            return False, False, None, None
        input_tokens = self.tokenizer.encode(self.build_input(prefix, suffix), return_tensors='pt')\
            .repeat(min(num_samples, self.batch_size), 1)
        input_tokens = input_tokens.to(self.device)
        with torch.no_grad():
            if use_max_length:
                raw_o = self.model.generate(input_tokens,
                                            max_length=len(input_tokens[0])
                                                       + len(self.tokenizer.encode(buggy, return_tensors='pt')[0])
                                                       + 100,
                                            do_sample=do_sample,
                                            output_scores=True,
                                            return_dict_in_generate=True,
                                            temperature=0.8,
                                            top_p=0.95)
            else:
                raw_o = self.model.generate(input_tokens,
                                            max_length=len(input_tokens[0]) + 50,
                                            do_sample=do_sample,
                                            output_scores=True,
                                            return_dict_in_generate=True,
                                            temperature=0.8,
                                            top_p=0.95)


            entropies = []
            if 'Salesforce' in self.pretrained:
                outputs = self.tokenizer.batch_decode(raw_o.sequences, skip_special_tokens=True)
                neg_logs = -torch.log(torch.stack(raw_o.scores, dim=1).softmax(-1))
                neg_logs = torch.gather(neg_logs, 2, raw_o.sequences[:, 1:, None]).squeeze(-1)
                t_outputs = []
                for index, output in enumerate(outputs):
                    if self.END_ID in raw_o.sequences[index, 1:]:
                        min_index = (raw_o.sequences[index, 1:] == self.END_ID).nonzero(as_tuple=True)[0][0].cpu().item()
                        infill_index = (raw_o.sequences[index, 1:] == self.infill_ID).nonzero(as_tuple=True)[0][0].cpu().item()
                        entropies.append(
                            (neg_logs[index, infill_index+1:min_index].sum(-1).cpu().item() / len(neg_logs[index, infill_index+1:min_index]),
                             neg_logs[index, infill_index+1:min_index].sum(-1).cpu().item()))
                        t_outputs.append(output)
                outputs = t_outputs
            elif 'facebook' in self.pretrained:
                gen_sequences = raw_o.sequences[:, len(input_tokens[0]):]
                neg_logs = -torch.log(torch.stack(raw_o.scores, dim=1).softmax(-1))
                neg_logs = torch.gather(neg_logs, 2, gen_sequences[:, :, None]).squeeze(-1)
                outputs = self.tokenizer.batch_decode(gen_sequences, clean_up_tokenization_spaces=False)
                t_outputs = []
                # print(outputs[0])
                for index, output in enumerate(outputs):
                    # print(output)
                    if output.startswith(self.BOS):
                        # print(output)
                        assert False
                    # This is a hack to stop the prefix and suffix generated by the model to be wrong
                    # TODO: contact the developer to double check this
                    # output = output[len(self.build_input(prefix, suffix)):]
                    if self.EOM not in output:
                        print("EOM not in")
                        continue
                        # only for infilling entire function, sometimes it does not stop in time
                        # for eof_string in global_infill_stops:
                        #     if eof_string in output:
                        #         output = output[:output.index(eof_string)]
                                # return True, True, output.strip()
                        # return False, True, ""
                    output = output[:output.index(self.EOM)]
                    if self.META_FILE in output:  # removes META file token that is sometimes generated
                        output = output[:output.index(self.META_FILE)]
                    print(output)
                    # exit()
                    min_index = (gen_sequences[index] == self.EOM_ID).nonzero(as_tuple=True)[0][0].cpu().item()
                    entropies.append(
                        (neg_logs[index, :min_index].sum(-1).cpu().item() / (min_index + 1),
                         neg_logs[index, :min_index].sum(-1).cpu().item()))
                    t_outputs.append(output)
                outputs = t_outputs

        return True, True, outputs, entropies
