# import os
# from cerebras.cloud.sdk import Cerebras

# client = Cerebras(
#     # This is the default and can be omitted
#     api_key=os.environ.get("CEREBRAS_API_KEY")
# )

# stream = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": ""
#         }
#     ],
#     model="gpt-oss-120b",
#     stream=True,
#     max_completion_tokens=65536,
#     temperature=1,
#     top_p=1
# )

my_dict = {'apple': 1, 'banana': 2, 'cherry': 3}

# Enumerate through both keys and values
for index, (key, value) in enumerate(my_dict.items()):
    print(f"Index: {index}, Key: {key}, Value: {value}")

# Enumerate only through keys (less common with dictionaries)
print("\nEnumerate through keys only:")
for index, key in enumerate(my_dict): # By default, iterating a dict iterates its keys
    print(f"Index: {index}, Key: {key}")