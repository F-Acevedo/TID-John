import openai

openai.api_key = "xd"

def openaiAPI(df, word_1, word_2):
    messages = []
    
    # Add a message for each text in the dataframe
    for text in df["Text"]:
        messages.append({"role": "user", "content": text})
    
    messages.append({"role": "user", "content": f"Ahora enlista las relaciones entre {word_1} y {word_2} luego explica cada relacion."})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    print(response['choices'][0]['message']['content'])












