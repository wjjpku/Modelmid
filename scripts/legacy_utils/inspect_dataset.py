from datasets import load_dataset
ds = load_dataset("math-ai/StackMathQA", "stackmathqa100k")
print(ds['train'][0])
