import re

text = "Hi how are you<START>I am good<END><SUM><START>great!<END><SUM>See you"

# Use a regex that captures the <START>...<END><SUM> segments.
# The non-greedy quantifier .*? ensures we capture the shortest possible substring
# from <START> to <END><SUM>.
pattern = r'(<START>.*?<END><SUM>)'

# re.split with a capturing group will include the matched segments in the result
parts = re.split(pattern, text)

# Filter out any empty strings that might appear due to splitting
parts = [part for part in parts if part != ""]

print(parts)
# Expected output:
# ["Hi how are you", "<START>I am good<END><SUM>", "<START>great!<END><SUM>", "See you"]
