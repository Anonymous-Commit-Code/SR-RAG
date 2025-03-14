
def create_prompt(input: list, prompt_file: str):        
  """
  读取prompt文件, 并且填充参数

  :param input: 给prompt里填充的参数。
  :param prompt_file: prompt对应的文件
  :return: 填充后的prompt。
  """
  with open(prompt_file, 'r') as f:
    template = f.read()
  prompt = template.split("===prompt===")[1]
  for i, v in enumerate(input):
    prompt = prompt.replace(f"!<INPUT {i}>!", str(v))
  return prompt

