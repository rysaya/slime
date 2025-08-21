def empty_template(prompt, tokenizer, tools=None):
    return prompt


def dpsk_zero_template(prompt, tokenizer, tools=None):
    prompt_template = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. \
The assistant first thinks about the reasoning process in the mind and then provides the user \
with the answer. The reasoning process and answer are enclosed within <think> </think> and \
<answer> </answer> tags, and put your final answer within \\boxed{} respectively, i.e., <think> reasoning process here </think> \
<answer> answer description here. Final answer: \\boxed{...} </answer>\n"""
    return prompt_template + "User: " + prompt + "\n Assistant: "


def get_chat_template(template_name):
    if template_name.lower() == "empty":
        return empty_template
    elif template_name.lower() == "dpsk_zero":
        return dpsk_zero_template
    else:
        raise ValueError(f"Unknown template name: {template_name}")
